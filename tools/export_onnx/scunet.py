"""Export SCUNet real-image denoisers (psnr, gan) to ONNX.

    uv run --group export python tools/export_onnx/scunet.py [--variant psnr|gan|all]

The network (config [4]*7, dim 64, head dim 32, window 8) is re-implemented below from
models/network_scunet.py at the pinned commit (Apache-2.0, Kai Zhang), without einops,
timm or thop, so no upstream code is fetched or run; the official weights load with
``strict=True``. Shifted-window masks are built from index arithmetic instead of in-place
indexing so the graph works for any size. Upstream pads to a multiple of 64 by edge
replication inside the network; here the app does that.

Checks: ONNX against PyTorch, and that each model moves a grainy real photo closer to the
clean original.

ONNX graph: input ``image`` (1x3xHxW RGB in [0,1]; H, W multiples of 64, dynamic) ->
output ``restored`` (same shape).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from colorizer.core import weights
from colorizer.core.runtime import CPU, create_session
from colorizer.models._onnx import onnx_path
from colorizer.models.scunet import CHECKPOINTS, MULTIPLE, ONNX

HEAD_DIM, WINDOW = 32, 8


class WMSA(nn.Module):
    """(Shifted) window multi-head self-attention, on b h w c tensors."""

    def __init__(self, dim: int, shifted: bool) -> None:
        super().__init__()
        self.n_heads, self.shifted = dim // HEAD_DIM, shifted
        self.embedding_layer = nn.Linear(dim, 3 * dim)
        self.relative_position_params = nn.Parameter(
            torch.zeros(self.n_heads, 2 * WINDOW - 1, 2 * WINDOW - 1)
        )
        self.linear = nn.Linear(dim, dim)
        cord = torch.tensor([[i, j] for i in range(WINDOW) for j in range(WINDOW)])
        relation = cord[:, None, :] - cord[None, :, :] + WINDOW - 1
        self.register_buffer("relation", relation, persistent=False)

    def mask(self, hw: int, ww: int) -> torch.Tensor:
        """Upstream generate_mask: True where a query and key come from different
        sub-windows of a wrapped-around (last row/column) window. (hw*ww, p*p, p*p)."""
        p, s = WINDOW, WINDOW - WINDOW // 2
        i = torch.arange(p)
        a = i[:, None] < s  # p1 / p2 in the first part
        b = i[None, :] < s  # p3 / p4 in the first part
        cross = a != b  # (p, p): one index in each part
        last_row = (torch.arange(hw) == hw - 1)[:, None, None, None, None, None]
        last_col = (torch.arange(ww) == ww - 1)[None, :, None, None, None, None]
        rows = cross[:, None, :, None]  # p1, p3
        cols = cross[None, :, None, :]  # p2, p4
        m = (last_row & rows[None, None]) | (last_col & cols[None, None])
        return m.reshape(hw * ww, p * p, p * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        p, nh = WINDOW, self.n_heads
        if self.shifted:
            x = torch.roll(x, shifts=(-(p // 2), -(p // 2)), dims=(1, 2))
        hw, ww = h // p, w // p
        x = x.view(b, hw, p, ww, p, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hw * ww, p * p, c)
        qkv = self.embedding_layer(x).view(b, hw * ww, p * p, 3 * nh, HEAD_DIM)
        qkv = qkv.permute(3, 0, 1, 2, 4)  # 3h b nw np c
        q, k, v = qkv[:nh], qkv[nh : 2 * nh], qkv[2 * nh :]
        sim = q @ k.transpose(-1, -2) * HEAD_DIM**-0.5
        rel = self.relative_position_params[:, self.relation[..., 0], self.relation[..., 1]]
        sim = sim + rel[:, None, None]
        if self.shifted:
            sim = sim.masked_fill(self.mask(hw, ww)[None, None], float("-inf"))
        out = F.softmax(sim, dim=-1) @ v  # h b nw np c
        out = out.permute(1, 2, 3, 0, 4).reshape(b, hw * ww, p * p, nh * HEAD_DIM)
        out = self.linear(out)
        out = out.view(b, hw, ww, p, p, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)
        if self.shifted:
            out = torch.roll(out, shifts=(p // 2, p // 2), dims=(1, 2))
        return out


class Block(nn.Module):
    def __init__(self, dim: int, shifted: bool) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.msa = WMSA(dim, shifted)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.msa(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ConvTransBlock(nn.Module):
    def __init__(self, dim: int, shifted: bool) -> None:
        super().__init__()
        self.half = dim
        self.trans_block = Block(dim, shifted)
        self.conv1_1 = nn.Conv2d(2 * dim, 2 * dim, 1)
        self.conv1_2 = nn.Conv2d(2 * dim, 2 * dim, 1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.half, self.half), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))


def stage(dim: int, n: int) -> list[nn.Module]:
    # Upstream alternates W / SW (every resolution is larger than the window at 256 input).
    return [ConvTransBlock(dim, shifted=i % 2 == 1) for i in range(n)]


class SCUNet(nn.Module):
    def __init__(self, dim: int = 64, n: int = 4) -> None:
        super().__init__()
        self.m_head = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(
            *stage(dim // 2, n), nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)
        )
        self.m_down2 = nn.Sequential(
            *stage(dim, n), nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)
        )
        self.m_down3 = nn.Sequential(
            *stage(2 * dim, n), nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)
        )
        self.m_body = nn.Sequential(*stage(4 * dim, n))
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), *stage(2 * dim, n)
        )
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), *stage(dim, n)
        )
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), *stage(dim // 2, n)
        )
        self.m_tail = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1, bias=False))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


def build(variant: str) -> SCUNet:
    net = SCUNet()
    state = torch.load(weights.ensure(CHECKPOINTS[variant]), map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("params", state), strict=True)
    return net.eval()


def grainy_photo() -> tuple[np.ndarray, np.ndarray]:
    """scikit-image's "camera" photo (256x256) and a copy with film-like gray grain."""
    from skimage import data

    clean = data.camera()[:256, 128:384].astype(np.float32) / 255.0
    grain = np.random.default_rng(0).normal(0, 0.06, clean.shape).astype(np.float32)
    return clean, np.clip(clean + grain, 0, 1)


def export(variant: str, out: Path) -> None:
    net = build(variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    k = MULTIPLE
    h, w = torch.export.Dim("h", min=1, max=128), torch.export.Dim("w", min=1, max=128)
    torch.onnx.export(
        net,
        (torch.rand(1, 3, 128, 192),),
        str(out),
        input_names=["image"],
        output_names=["restored"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"x0": {2: k * h, 3: k * w}},
    )

    session, _ = create_session(out, CPU)
    clean, grainy = grainy_photo()
    err = 0.0
    for crop in (grainy, grainy[:128, :192]):  # square and non-square window grids
        x = torch.from_numpy(np.repeat(crop[None, None], 3, axis=1).copy())
        with torch.no_grad():
            expected = net(x).numpy()
        (got,) = session.run(None, {"image": x.numpy()})
        err = max(err, float(np.abs(got - expected).max()))
    if err > 1e-3:
        raise SystemExit(f"{variant}: ONNX differs from PyTorch (max abs err {err:.4g})")
    (restored,) = session.run(None, {"image": np.repeat(grainy[None, None], 3, axis=1)})
    before = float(np.mean((grainy - clean) ** 2))
    after = float(np.mean((np.clip(restored[0].mean(axis=0), 0, 1) - clean) ** 2))
    if after >= before:
        raise SystemExit(f"{variant}: grainy photo got worse (MSE {before:.5f} -> {after:.5f})")
    logging.info("wrote %s (max abs err %.2e; MSE %.5f -> %.5f)", out, err, before, after)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--variant", choices=[*ONNX, "all"], default="all")
    args = parser.parse_args()
    for variant in ONNX if args.variant == "all" else [args.variant]:
        export(variant, onnx_path(ONNX[variant]))


if __name__ == "__main__":
    main()
