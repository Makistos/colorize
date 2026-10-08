"""Export CodeFormer (face restoration) to ONNX.

    uv run --group export python tools/export_onnx/codeformer.py

The network is re-implemented below from basicsr/archs/codeformer_arch.py and
vqgan_arch.py at the pinned commit (S-Lab License 1.0, non-commercial; Shangchen Zhou), so
no upstream code is fetched or run. The official weights (``params_ema``) load with
``strict=True``, which checks every layer. Changes that keep the maths identical:
- the ``if w > 0`` branch is dropped (with w = 0 the fusion adds exactly 0), so the
  fidelity weight can be a graph input;
- the codebook lookup uses argmax + embedding instead of top-1 of the softmax + one-hot
  matmul.

ONNX graph: inputs ``face`` (1x3x512x512 RGB in [-1, 1], an aligned face) and ``fidelity``
(float32 [1], 0..1) -> output ``restored`` (1x3x512x512 in [-1, 1]); adain=True as in
upstream inference.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from colorizer.core import weights
from colorizer.core.runtime import CPU, create_session
from colorizer.models._onnx import onnx_path
from colorizer.models.codeformer import CHECKPOINT, FACE_SIZE, ONNX


def normalize(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, ch, eps=1e-6, affine=True)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.norm1, self.conv1 = normalize(cin), nn.Conv2d(cin, cout, 3, 1, 1)
        self.norm2, self.conv2 = normalize(cout), nn.Conv2d(cout, cout, 3, 1, 1)
        self.conv_out = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv1(swish(self.norm1(x_in)))
        x = self.conv2(swish(self.norm2(x)))
        return x + (self.conv_out(x_in) if self.conv_out is not None else x_in)


class AttnBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = normalize(ch)
        self.q, self.k, self.v = nn.Conv2d(ch, ch, 1), nn.Conv2d(ch, ch, 1), nn.Conv2d(ch, ch, 1)
        self.proj_out = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, hh, ww = q.shape
        q = q.reshape(b, c, hh * ww).permute(0, 2, 1)
        attn = F.softmax(torch.bmm(q, k.reshape(b, c, hh * ww)) * (int(c) ** -0.5), dim=2)
        h = torch.bmm(v.reshape(b, c, hh * ww), attn.permute(0, 2, 1)).reshape(b, c, hh, ww)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


NF, CH_MULT, RES_BLOCKS, ATTN_RES, EMB_DIM, CODEBOOK = 64, (1, 2, 2, 4, 4, 8), 2, (16,), 256, 1024


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv2d(3, NF, 3, 1, 1)]
        res, in_mult = FACE_SIZE, (1, *CH_MULT)
        for i, mult in enumerate(CH_MULT):
            cin, cout = NF * in_mult[i], NF * mult
            for _ in range(RES_BLOCKS):
                blocks.append(ResBlock(cin, cout))
                cin = cout
                if res in ATTN_RES:
                    blocks.append(AttnBlock(cin))
            if i != len(CH_MULT) - 1:
                blocks.append(Downsample(cin))
                res //= 2
        blocks += [ResBlock(cin, cin), AttnBlock(cin), ResBlock(cin, cin)]
        blocks += [normalize(cin), nn.Conv2d(cin, EMB_DIM, 3, 1, 1)]
        self.blocks = nn.ModuleList(blocks)


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cin = NF * CH_MULT[-1]
        res = FACE_SIZE // 2 ** (len(CH_MULT) - 1)
        blocks: list[nn.Module] = [nn.Conv2d(EMB_DIM, cin, 3, 1, 1)]
        blocks += [ResBlock(cin, cin), AttnBlock(cin), ResBlock(cin, cin)]
        for i in reversed(range(len(CH_MULT))):
            cout = NF * CH_MULT[i]
            for _ in range(RES_BLOCKS):
                blocks.append(ResBlock(cin, cout))
                cin = cout
                if res in ATTN_RES:
                    blocks.append(AttnBlock(cin))
            if i != 0:
                blocks.append(Upsample(cin))
                res *= 2
        blocks += [normalize(cin), nn.Conv2d(cin, 3, 3, 1, 1)]
        self.blocks = nn.ModuleList(blocks)


class VectorQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(CODEBOOK, EMB_DIM)


class TransformerSALayer(nn.Module):
    def __init__(self, dim: int, nhead: int, dim_mlp: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=0.0)
        self.linear1, self.linear2 = nn.Linear(dim, dim_mlp), nn.Linear(dim_mlp, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, tgt: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        t = self.norm1(tgt)
        q = k = t + pos
        tgt = tgt + self.self_attn(q, k, value=t)[0]
        return tgt + self.linear2(F.gelu(self.linear1(self.norm2(tgt))))


class FuseSFT(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.encode_enc = ResBlock(2 * ch, ch)

        def branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ch, ch, 3, padding=1),
            )

        self.scale, self.shift = branch(), branch()

    def forward(self, enc: torch.Tensor, dec: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        enc = self.encode_enc(torch.cat([enc, dec], dim=1))
        return dec + w * (dec * self.scale(enc) + self.shift(enc))


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    b, c = feat.shape[:2]
    std = (feat.view(b, c, -1).var(dim=2) + eps).sqrt().view(b, c, 1, 1)
    return feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1), std


def adain(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    s_mean, s_std = calc_mean_std(style)
    c_mean, c_std = calc_mean_std(content)
    return (content - c_mean) / c_std * s_std + s_mean


class CodeFormer(nn.Module):
    CONNECT = ("32", "64", "128", "256")
    CHANNELS: ClassVar[dict[str, int]] = {
        "16": 512,
        "32": 256,
        "64": 256,
        "128": 128,
        "256": 128,
        "512": 64,
    }
    FUSE_ENCODER: ClassVar[dict[str, int]] = {
        "512": 2,
        "256": 5,
        "128": 8,
        "64": 11,
        "32": 14,
        "16": 18,
    }
    FUSE_GENERATOR: ClassVar[dict[str, int]] = {
        "16": 6,
        "32": 9,
        "64": 12,
        "128": 15,
        "256": 18,
        "512": 21,
    }

    def __init__(self, dim: int = 512, n_head: int = 8, n_layers: int = 9) -> None:
        super().__init__()
        self.encoder, self.quantize, self.generator = Encoder(), VectorQuantizer(), Generator()
        self.position_emb = nn.Parameter(torch.zeros(256, dim))
        self.feat_emb = nn.Linear(256, dim)
        self.ft_layers = nn.Sequential(
            *[TransformerSALayer(dim, n_head, dim * 2) for _ in range(n_layers)]
        )
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, CODEBOOK, bias=False))
        self.fuse_convs_dict = nn.ModuleDict({s: FuseSFT(self.CHANNELS[s]) for s in self.CONNECT})

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        enc_feats = {}
        enc_at = {self.FUSE_ENCODER[s]: s for s in self.CONNECT}
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in enc_at:
                enc_feats[enc_at[i]] = x
        lq_feat = x
        pos = self.position_emb.unsqueeze(1)  # (hw) 1 c, batch size 1
        query = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        for layer in self.ft_layers:
            query = layer(query, pos)
        logits = self.idx_pred_layer(query).permute(1, 0, 2)  # 1 (hw) n
        idx = logits.argmax(dim=2)  # = top-1 of the softmax
        quant = self.quantize.embedding(idx).view(1, 16, 16, EMB_DIM).permute(0, 3, 1, 2)
        x = adain(quant, lq_feat)
        gen_at = {self.FUSE_GENERATOR[s]: s for s in self.CONNECT}
        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in gen_at:
                s = gen_at[i]
                x = self.fuse_convs_dict[s](enc_feats[s], x, w)
        return x


def build() -> CodeFormer:
    net = CodeFormer()
    ckpt = torch.load(weights.ensure(CHECKPOINT), map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["params_ema"], strict=True)
    return net.eval()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    net = build()
    out = onnx_path(ONNX)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        (torch.rand(1, 3, FACE_SIZE, FACE_SIZE) * 2 - 1, torch.tensor([0.5])),
        str(out),
        input_names=["face", "fidelity"],
        output_names=["restored"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )

    session, _ = create_session(out, CPU)
    from skimage import data

    face = torch.from_numpy(data.astronaut()[:FACE_SIZE, :FACE_SIZE]).float()
    face = (face.permute(2, 0, 1)[None] / 127.5 - 1).contiguous()
    err = 0.0
    for w in (0.0, 0.7, 1.0):
        fidelity = torch.tensor([w])
        with torch.no_grad():
            expected = net(face, fidelity).numpy()
        (got,) = session.run(None, {"face": face.numpy(), "fidelity": fidelity.numpy()})
        err = max(err, float(np.abs(got - expected).max()))
    if err > 1e-2:
        raise SystemExit(f"ONNX differs from PyTorch (max abs err {err:.4g})")
    logging.info("wrote %s (max abs err %.2e)", out, err)


if __name__ == "__main__":
    main()
