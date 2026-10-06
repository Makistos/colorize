"""Export Zhang ECCV16 and SIGGRAPH17 to ONNX.

    uv run --group export python tools/export_onnx/zhang.py [--model eccv16|siggraph17|all]

Downloads the official checkpoints (SHA256-verified) and rebuilds the networks from
richzhang/colorization ``colorizers/{eccv16,siggraph17}.py``; weights load with strict=True.
Normalisation is baked into the graphs:

- ECCV16: ``L`` (1x1x256x256, L in [0,100]) -> ``ab`` (1x2x256x256).
- SIGGRAPH17: ``L`` (1x1xHxW), ``hint_ab`` (1x2xHxW, Lab ab), ``hint_mask`` (1x1xHxW, 1 where a
  hint is given) -> ``ab`` (1x2xHxW); H and W multiples of 8, dynamic.
"""

from __future__ import annotations

import argparse
import logging
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn

from colorizer.core import weights
from colorizer.models.zhang import (
    ECCV16_CHECKPOINT,
    ECCV16_ONNX,
    ECCV16_SIZE,
    SIGGRAPH17_CHECKPOINT,
    SIGGRAPH17_ONNX,
    onnx_path,
)

# The released SIGGRAPH17 weights take the hint mask uncentered (0 = no hint, 1 = hint), as
# in upstream colorizers/siggraph17.py. Centering it by 0.5 (as colorization-pytorch's
# training code does) gives a teal cast and hints that do not propagate; checked visually.
MASK_CENT = 0.0


def _block(chans: list[int], strides: list[int] | None = None, dilation: int = 1) -> nn.Sequential:
    strides = strides or [1] * (len(chans) - 1)
    layers: list[nn.Module] = []
    for cin, cout, stride in zip(chans[:-1], chans[1:], strides, strict=True):
        layers += [
            nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation),
            nn.ReLU(True),
        ]
    layers.append(nn.BatchNorm2d(chans[-1]))
    return nn.Sequential(*layers)


class ECCVGenerator(nn.Module):
    """Same module layout as the upstream implementation so the state dict loads strictly."""

    def __init__(self) -> None:
        super().__init__()
        self.model1 = _block([1, 64, 64], [1, 2])
        self.model2 = _block([64, 128, 128], [1, 2])
        self.model3 = _block([128, 256, 256, 256], [1, 1, 2])
        self.model4 = _block([256, 512, 512, 512])
        self.model5 = _block([512, 512, 512, 512], dilation=2)
        self.model6 = _block([512, 512, 512, 512], dilation=2)
        self.model7 = _block([512, 512, 512, 512])
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 313, 1),
        )
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, 1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        x = (L - 50.0) / 100.0
        for block in (
            self.model1,
            self.model2,
            self.model3,
            self.model4,
            self.model5,
            self.model6,
            self.model7,
            self.model8,
        ):
            x = block(x)
        return self.upsample4(self.model_out(self.softmax(x))) * 110.0


def export_eccv16(out: Path) -> None:
    ckpt = weights.ensure(ECCV16_CHECKPOINT)
    net = ECCVGenerator()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    net.eval()
    dummy = torch.rand(1, 1, ECCV16_SIZE, ECCV16_SIZE) * 100.0
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        (dummy,),
        str(out),
        input_names=["L"],
        output_names=["ab"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    _check(net, out, [{"L": dummy}])


def export_siggraph17(out: Path) -> None:
    net = load_siggraph17()

    def inputs(h: int, w: int) -> dict[str, torch.Tensor]:
        mask = (torch.rand(1, 1, h, w) > 0.99).float()
        return {
            "L": torch.rand(1, 1, h, w) * 100,
            "hint_ab": torch.randn(1, 2, h, w) * 40 * mask,
            "hint_mask": mask,
        }

    example = inputs(256, 256)
    h, w = torch.export.Dim("h", min=4, max=256), torch.export.Dim("w", min=4, max=256)
    shape = {2: 8 * h, 3: 8 * w}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        tuple(example.values()),
        str(out),
        input_names=list(example),
        output_names=["ab"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"L": shape, "ab": shape, "mask": shape},
    )
    _check(net, out, [example, inputs(320, 448)])


def _check(net: nn.Module, out: Path, cases: list[dict[str, torch.Tensor]]) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    err = 0.0
    for feeds in cases:
        with torch.no_grad():
            expected = net(*feeds.values()).numpy()
        (got,) = sess.run(None, {k: v.numpy() for k, v in feeds.items()})
        err = max(err, float(np.abs(got - expected).max()))
    if err > 1e-2:
        raise SystemExit(f"{out.name}: ONNX output differs from PyTorch (max abs err {err:.4g})")
    logging.info("wrote %s (max abs err vs PyTorch %.2e)", out, err)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", choices=["eccv16", "siggraph17", "all"], default="all")
    args = parser.parse_args()
    if args.model in ("eccv16", "all"):
        export_eccv16(onnx_path(ECCV16_ONNX))
    if args.model in ("siggraph17", "all"):
        export_siggraph17(onnx_path(SIGGRAPH17_ONNX))


def _convs(chans: list[int], dilation: int = 1, norm: bool = True) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for cin, cout in pairwise(chans):
        layers += [nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation), nn.ReLU(True)]
    if norm:
        layers.append(nn.BatchNorm2d(chans[-1]))
    return layers


class SIGGRAPHGenerator(nn.Module):
    """richzhang/colorization ``colorizers/siggraph17.py`` (module names match the weights).

    Inputs are raw: L in [0,100], hint ab in Lab units, mask 1 where a hint is given.
    """

    def __init__(self, classes: int = 529) -> None:
        super().__init__()
        self.model1 = nn.Sequential(*_convs([4, 64, 64]))
        self.model2 = nn.Sequential(*_convs([64, 128, 128]))
        self.model3 = nn.Sequential(*_convs([128, 256, 256, 256]))
        self.model4 = nn.Sequential(*_convs([256, 512, 512, 512]))
        self.model5 = nn.Sequential(*_convs([512, 512, 512, 512], dilation=2))
        self.model6 = nn.Sequential(*_convs([512, 512, 512, 512], dilation=2))
        self.model7 = nn.Sequential(*_convs([512, 512, 512, 512]))
        self.model8up = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1))
        self.model3short8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1))
        self.model8 = nn.Sequential(nn.ReLU(True), *_convs([256, 256, 256]))
        self.model9up = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1))
        self.model2short9 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1))
        self.model9 = nn.Sequential(nn.ReLU(True), *_convs([128, 128]))
        self.model10up = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1))
        self.model1short10 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1))
        self.model10 = nn.Sequential(
            nn.ReLU(True), nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(negative_slope=0.2)
        )
        self.model_class = nn.Sequential(nn.Conv2d(256, classes, 1))  # unused at inference
        self.model_out = nn.Sequential(nn.Conv2d(128, 2, 1), nn.Tanh())

    def forward(self, L: torch.Tensor, ab: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([(L - 50.0) / 100.0, ab / 110.0, mask - MASK_CENT], dim=1)
        conv1_2 = self.model1(x)
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv7_3 = self.model7(self.model6(self.model5(conv4_3)))
        conv8_3 = self.model8(self.model8up(conv7_3) + self.model3short8(conv3_3))
        conv9_3 = self.model9(self.model9up(conv8_3) + self.model2short9(conv2_2))
        conv10_2 = self.model10(self.model10up(conv9_3) + self.model1short10(conv1_2))
        return self.model_out(conv10_2) * 110.0


def load_siggraph17() -> SIGGRAPHGenerator:
    net = SIGGRAPHGenerator()
    state = torch.load(weights.ensure(SIGGRAPH17_CHECKPOINT), map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    return net.eval()


if __name__ == "__main__":
    main()
