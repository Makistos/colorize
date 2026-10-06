"""Export DeOldify (artistic, stable) to ONNX without fastai.

    uv run --group export python tools/export_onnx/deoldify.py [--variant artistic|stable|all]

Rebuilds the fastai v1 ``DynamicUnetDeep`` (artistic, ResNet34) and ``DynamicUnetWide``
(stable, ResNet101) generators from jantic/DeOldify ``deoldify/unet.py`` +
``fastai/layers.py`` with identical module paths, so the official checkpoints load with
``strict=True``. Spectral/weight norms are folded into plain weights before export.

ONNX graph: input ``gray`` (1x3xHxW, gray RGB in [0,1], H and W multiples of 16, dynamic)
-> output ``rgb`` (1x3xHxW, colorized RGB in [0,1]). ImageNet (de)normalisation is baked in.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.utils import remove_spectral_norm, remove_weight_norm, spectral_norm, weight_norm

from colorizer.core import weights
from colorizer.models.deoldify import VARIANTS, onnx_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- fastai v1 / DeOldify layers (structure must match checkpoint keys) ---------------------


def conv_layer(ni, nf, ks=3, norm="batch", use_activ=True, self_attention=False, extra_bn=False):
    """fastai ``conv_layer`` / DeOldify ``custom_conv_layer``."""
    bn = norm == "batch" or extra_bn
    conv = nn.Conv2d(ni, nf, ks, padding=(ks - 1) // 2, bias=not bn)
    if norm == "weight":
        conv = weight_norm(conv)
    elif norm == "spectral":
        conv = spectral_norm(conv)
    layers: list[nn.Module] = [conv]
    if use_activ:
        layers.append(nn.ReLU(True))
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


class SelfAttention(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.query = spectral_norm(nn.Conv1d(n, n // 8, 1, bias=False))
        self.key = spectral_norm(nn.Conv1d(n, n // 8, 1, bias=False))
        self.value = spectral_norm(nn.Conv1d(n, n, 1, bias=False))
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class PixelShuffleICNR(nn.Module):
    """Both fastai ``PixelShuffle_ICNR`` and DeOldify ``CustomPixelShuffle_ICNR``.

    Upstream assigns ``self.blur = nn.AvgPool2d(...)``, shadowing the ``blur`` flag, so the
    blur is always applied. Reproduced here.
    """

    def __init__(self, ni, nf, scale=2, norm="weight", extra_bn=False):
        super().__init__()
        self.conv = conv_layer(
            ni, nf * scale**2, ks=1, norm=norm, use_activ=False, extra_bn=extra_bn
        )
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.blur(self.pad(self.shuf(self.relu(self.conv(x)))))


class UnetBlockDeep(nn.Module):
    def __init__(self, up_in_c, x_in_c, final_div, self_attention, nf_factor):
        super().__init__()
        self.shuf = PixelShuffleICNR(up_in_c, up_in_c // 2, norm="spectral", extra_bn=True)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = int((ni if final_div else ni // 2) * nf_factor)
        self.conv1 = conv_layer(ni, nf, norm="spectral", extra_bn=True)
        self.conv2 = conv_layer(
            nf, nf, norm="spectral", extra_bn=True, self_attention=self_attention
        )
        self.out_channels = nf

    def forward(self, up_in, s):
        up_out = self.shuf(up_in)
        # Upstream resizes only on mismatch; doing it unconditionally (a no-op when sizes
        # agree) keeps the exported graph valid for sizes that are not multiples of 32.
        up_out = F.interpolate(up_out, s.shape[-2:], mode="nearest")
        return self.conv2(self.conv1(F.relu(torch.cat([up_out, self.bn(s)], dim=1))))


class UnetBlockWide(nn.Module):
    def __init__(self, up_in_c, x_in_c, n_out, self_attention):
        super().__init__()
        up_out = x_out = n_out // 2
        self.shuf = PixelShuffleICNR(up_in_c, up_out, norm="spectral", extra_bn=True)
        self.bn = nn.BatchNorm2d(x_in_c)
        self.conv = conv_layer(
            up_out + x_in_c, x_out, norm="spectral", extra_bn=True, self_attention=self_attention
        )
        self.out_channels = x_out

    def forward(self, up_in, s):
        up_out = self.shuf(up_in)
        # Upstream resizes only on mismatch; doing it unconditionally (a no-op when sizes
        # agree) keeps the exported graph valid for sizes that are not multiples of 32.
        up_out = F.interpolate(up_out, s.shape[-2:], mode="nearest")
        return self.conv(F.relu(torch.cat([up_out, self.bn(s)], dim=1)))


class ResBlock(nn.Module):
    """fastai ``res_block(dense=False)`` as a ``SequentialEx`` (keys ``layers.N``)."""

    def __init__(self, nf):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                conv_layer(nf, nf, norm="spectral"),
                conv_layer(nf, nf, norm="spectral"),
                nn.Identity(),  # MergeLayer
            ]
        )

    def forward(self, x):
        return self.layers[1](self.layers[0](x)) + x


# Encoder children whose outputs feed the decoder (fastai ``_get_sfs_idxs`` on ResNets):
# relu (1/2), layer1 (1/4), layer2 (1/8), layer3 (1/16); deepest first.
SKIP_IDXS = (6, 5, 4, 2)


class DeOldifyUnet(nn.Module):
    def __init__(self, arch: str, wide: bool, nf_factor: float):
        super().__init__()
        resnet = getattr(torchvision.models, arch)(weights=None)
        encoder = nn.Sequential(*list(resnet.children())[:-2])
        with torch.no_grad():
            feats, x = {}, torch.zeros(1, 3, 256, 256)
            for i, layer in enumerate(encoder):
                x = layer(x)
                feats[i] = x.shape[1]
        ni = x.shape[1]
        middle = nn.Sequential(
            conv_layer(ni, ni * 2, norm="spectral", extra_bn=True),
            conv_layer(ni * 2, ni, norm="spectral", extra_bn=True),
        )
        layers: list[nn.Module] = [encoder, nn.BatchNorm2d(ni), nn.ReLU(), middle]
        up_c = ni
        for i, idx in enumerate(SKIP_IDXS):
            not_final = i != len(SKIP_IDXS) - 1
            sa = i == len(SKIP_IDXS) - 3
            if wide:
                nf = int(512 * nf_factor)
                block: nn.Module = UnetBlockWide(up_c, feats[idx], nf if not_final else nf // 2, sa)
            else:
                block = UnetBlockDeep(up_c, feats[idx], not_final, sa, nf_factor)
            layers.append(block)
            up_c = block.out_channels
        layers += [
            PixelShuffleICNR(up_c, up_c, norm="weight"),
            nn.Identity(),  # MergeLayer(dense=True): concat the network input
            ResBlock(up_c + 3),
            conv_layer(up_c + 3, 3, ks=1, norm="spectral", use_activ=False),
            nn.Identity(),  # SigmoidRange(-3, 3)
        ]
        self.layers = nn.ModuleList(layers)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, gray):
        inp = (gray - self.mean) / self.std
        skips = {}
        x = inp
        for i, layer in enumerate(self.layers[0]):
            x = layer(x)
            if i in SKIP_IDXS:
                skips[i] = x
        x = self.layers[3](self.layers[2](self.layers[1](x)))
        for block, idx in zip(self.layers[4:8], SKIP_IDXS, strict=True):
            x = block(x, skips[idx])
        x = self.layers[8](x)
        x = torch.cat([x, inp], dim=1)
        x = self.layers[11](self.layers[10](x))
        x = torch.sigmoid(x) * 6.0 - 3.0
        return (x * self.std + self.mean).clamp(0.0, 1.0)


ARCHS = {
    "artistic": ("resnet34", False, 1.5),
    "stable": ("resnet101", True, 2),
}


def fold_norms(model: nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "weight_orig"):
            remove_spectral_norm(m)
        elif hasattr(m, "weight_g"):
            remove_weight_norm(m)


def export(variant: str, out: Path) -> None:
    arch, wide, nf_factor = ARCHS[variant]
    ckpt = weights.ensure(VARIANTS[variant].checkpoint)
    # fastai Learner.save pickles {"model", "opt"}; the file is SHA256-pinned above.
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    net = DeOldifyUnet(arch, wide, nf_factor)
    net.load_state_dict(state, strict=True)
    net.eval()
    fold_norms(net)

    dummy = torch.rand(1, 3, 256, 256)
    out.parent.mkdir(parents=True, exist_ok=True)
    # In units of 16 px: 64..4096 px.
    h, w = torch.export.Dim("h", min=4, max=256), torch.export.Dim("w", min=4, max=256)
    torch.onnx.export(
        net,
        (dummy,),
        str(out),
        input_names=["gray"],
        output_names=["rgb"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"gray": {2: 16 * h, 3: 16 * w}},
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for shape in ((256, 256), (560, 400)):
        x = torch.rand(1, 3, *shape)
        with torch.no_grad():
            expected = net(x).numpy()
        (got,) = sess.run(None, {"gray": x.numpy()})
        err = float(np.abs(got - expected).max())
        if err > 1e-3:
            raise SystemExit(f"{variant}: ONNX differs from PyTorch at {shape} (max err {err:.4g})")
    logging.info("wrote %s (max abs err %.2e)", out, err)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=[*ARCHS, "all"], default="all")
    args = parser.parse_args()
    for variant in ARCHS if args.variant == "all" else [args.variant]:
        export(variant, onnx_path(VARIANTS[variant].onnx))


if __name__ == "__main__":
    main()
