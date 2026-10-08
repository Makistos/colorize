"""Export Real-ESRGAN realesr-general-x4v3 to ONNX, with denoise strength as an input.

    uv run --group export python tools/export_onnx/realesrgan.py

The network (SRVGGNetCompact, num_conv=32, PReLU, x4) is re-implemented below from
realesrgan/archs/srvgg_arch.py at the pinned commit (BSD-3-Clause, Xintao Wang) so basicsr
is not needed; the official weights load with ``strict=True``.

Upstream controls denoising with DNI: the weights of the general model (strong denoise)
and its "wdn" twin (weak denoise) are interpolated, ``d * general + (1 - d) * wdn``. The
exported graph holds both weight sets and does that interpolation itself, per layer, from a
``denoise`` input, so one static graph covers every strength on every execution provider.

ONNX graph: inputs ``image`` (1x3xHxW RGB in [0,1], dynamic) and ``denoise`` (float32 [1],
0..1) -> output ``upscaled`` (1x3x4Hx4W). Written to the cache dir.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from colorizer.core import weights
from colorizer.core.runtime import CPU, create_session
from colorizer.models._onnx import onnx_path
from colorizer.models.realesrgan import GENERAL_CHECKPOINT, NET_SCALE, ONNX, WDN_CHECKPOINT


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat: int = 64, num_conv: int = 32, upscale: int = NET_SCALE) -> None:
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class DNINet(nn.Module):
    """SRVGGNetCompact whose weights are ``d * strong + (1 - d) * weak`` for input ``d``."""

    def __init__(self, strong: SRVGGNetCompact, weak: SRVGGNetCompact) -> None:
        super().__init__()
        self.upscale = strong.upscale
        self.kinds = ["conv" if isinstance(m, nn.Conv2d) else "prelu" for m in strong.body]
        for name, value in strong.state_dict().items():
            self.register_buffer("s_" + name.replace(".", "_"), value.clone())
        for name, value in weak.state_dict().items():
            self.register_buffer("w_" + name.replace(".", "_"), value.clone())

    def param(self, name: str, d: torch.Tensor) -> torch.Tensor:
        key = name.replace(".", "_")
        strong: torch.Tensor = getattr(self, "s_" + key)
        weak: torch.Tensor = getattr(self, "w_" + key)
        return d * strong + (1 - d) * weak

    def forward(self, x: torch.Tensor, denoise: torch.Tensor) -> torch.Tensor:
        out = x
        for i, kind in enumerate(self.kinds):
            if kind == "conv":
                w = self.param(f"body.{i}.weight", denoise)
                out = F.conv2d(out, w, self.param(f"body.{i}.bias", denoise), padding=1)
            else:
                out = F.prelu(out, self.param(f"body.{i}.weight", denoise))
        out = F.pixel_shuffle(out, self.upscale)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def build(checkpoint: weights.WeightFile) -> SRVGGNetCompact:
    net = SRVGGNetCompact()
    state = torch.load(weights.ensure(checkpoint), map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("params", state), strict=True)
    return net.eval()


def export(net: nn.Module, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = torch.export.Dim("h", min=8, max=4096), torch.export.Dim("w", min=8, max=4096)
    torch.onnx.export(
        net,
        (torch.rand(1, 3, 64, 64), torch.tensor([0.5])),
        str(out),
        input_names=["image", "denoise"],
        output_names=["upscaled"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"x": {2: h, 3: w}, "denoise": None},
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    strong, weak = build(GENERAL_CHECKPOINT), build(WDN_CHECKPOINT)
    out = onnx_path(ONNX)
    export(DNINet(strong, weak).eval(), out)

    # Check against upstream-style DNI: a plain network loaded with interpolated weights.
    session, _ = create_session(out, CPU)
    err = 0.0
    x = torch.rand(1, 3, 48, 80)
    for d in (1.0, 0.5, 0.0):
        mixed = SRVGGNetCompact().eval()
        weak_state = weak.state_dict()
        mixed.load_state_dict(
            {k: d * v + (1 - d) * weak_state[k] for k, v in strong.state_dict().items()}
        )
        with torch.no_grad():
            expected = mixed(x).numpy()
        feeds = {"image": x.numpy(), "denoise": np.array([d], np.float32)}
        (got,) = session.run(None, feeds)
        err = max(err, float(np.abs(got - expected).max()))
    if err > 1e-3:
        raise SystemExit(f"ONNX differs from PyTorch (max abs err {err:.4g})")
    logging.info("wrote %s (max abs err %.2e)", out, err)


if __name__ == "__main__":
    main()
