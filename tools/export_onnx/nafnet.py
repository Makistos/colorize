"""Export NAFNet (deblur: GoPro-width64, denoise: SIDD-width64) to ONNX.

    uv run --group export python tools/export_onnx/nafnet.py [--variant deblur|denoise|all]

The network is re-implemented below from basicsr/models/archs/NAFNet_arch.py and
arch_util.py (LayerNorm2d) at the pinned commit (MIT, megvii-model), so no upstream code is
fetched or run; the official weights load with ``strict=True``, which checks every layer.
Upstream pads inputs to a multiple of 16 inside the network; here the app does that.

Checks: ONNX against PyTorch, and that each model moves a degraded image (blurred or noisy)
closer to the clean original.

ONNX graph: input ``image`` (1x3xHxW RGB in [0,1]; H, W multiples of 16, dynamic) ->
output ``restored`` (same shape).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from colorizer.core import weights
from colorizer.core.runtime import CPU, create_session
from colorizer.models._onnx import onnx_path
from colorizer.models.nafnet import MULTIPLE, VARIANTS


class LayerNorm2d(nn.Module):
    """Layer norm over channels (upstream LayerNormFunction.forward)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


def simple_gate(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=1)
    return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        dw = c * 2
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv4 = nn.Conv2d(c, c * 2, 1)
        self.conv5 = nn.Conv2d(c, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = simple_gate(self.conv2(self.conv1(self.norm1(inp))))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(simple_gate(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self, width: int, middle_blk_num: int, enc_blk_nums: tuple[int, ...],
        dec_blk_nums: tuple[int, ...],
    ) -> None:  # fmt: skip
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            up = nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            self.ups.append(up)
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            y = encoder(y)
            skips.append(y)
            y = down(y)
        y = self.middle_blks(y)
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1], strict=True):
            y = decoder(up(y) + skip)
        return self.ending(y) + x


def build(variant: str) -> NAFNet:
    v = VARIANTS[variant]
    net = NAFNet(64, v.middle_blk_num, v.enc_blk_nums, v.dec_blk_nums)
    state = torch.load(weights.ensure(v.checkpoint), map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("params", state), strict=True)
    return net.eval()


def test_image() -> np.ndarray:
    """A real grayscale photograph (scikit-image's bundled "camera"), HxW in [0, 1]."""
    from skimage import data

    return data.camera()[:256, 128:384].astype(np.float32) / 255.0


def degrade(variant: str, clean: np.ndarray) -> np.ndarray:
    if variant == "deblur":
        kernel = np.zeros((9, 9), np.float32)
        kernel[4, :] = 1 / 9  # horizontal motion blur
        return cv2.filter2D(clean, -1, kernel, borderType=cv2.BORDER_REFLECT)
    noise = np.random.default_rng(0).normal(0, 0.05, clean.shape).astype(np.float32)
    return np.clip(clean + noise, 0, 1)


def export(variant: str, out: Path) -> None:
    net = build(variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    k = MULTIPLE
    h, w = torch.export.Dim("h", min=1, max=512), torch.export.Dim("w", min=1, max=512)
    torch.onnx.export(
        net,
        (torch.rand(1, 3, 128, 128),),
        str(out),
        input_names=["image"],
        output_names=["restored"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"x": {2: k * h, 3: k * w}},
    )

    session, _ = create_session(out, CPU)
    clean = test_image()
    degraded = degrade(variant, clean)
    err = 0.0
    # Real-looking input: on uniform noise the network is ill-conditioned (float32 and
    # float64 PyTorch already differ by ~0.03), which says nothing about the export.
    for crop in (degraded, degraded[16:112, 32:192]):  # 256x256 and 96x160
        x = torch.from_numpy(np.repeat(crop[None, None], 3, axis=1).copy())
        with torch.no_grad():
            expected = net(x).numpy()
        (got,) = session.run(None, {"image": x.numpy()})
        err = max(err, float(np.abs(got - expected).max()))
    if err > 1e-3:
        raise SystemExit(f"{variant}: ONNX differs from PyTorch (max abs err {err:.4g})")

    (restored,) = session.run(None, {"image": np.repeat(degraded[None, None], 3, axis=1)})
    before = float(np.mean((degraded - clean) ** 2))
    after = float(np.mean((np.clip(restored[0].mean(axis=0), 0, 1) - clean) ** 2))
    if after >= before:
        raise SystemExit(f"{variant}: test image got worse (MSE {before:.5f} -> {after:.5f})")
    logging.info("wrote %s (max abs err %.2e; test MSE %.5f -> %.5f)", out, err, before, after)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = parser.parse_args()
    for variant in VARIANTS if args.variant == "all" else [args.variant]:
        export(variant, onnx_path(VARIANTS[variant].onnx))


if __name__ == "__main__":
    main()
