"""Export Zhang ECCV16 to ONNX.

    uv run --group export python tools/export_onnx/zhang.py [--out PATH]

Downloads the official checkpoint (SHA256-verified), rebuilds the network from
richzhang/colorization ``colorizers/eccv16.py`` and writes an ONNX graph with
input ``L`` (1x1x256x256, raw L in [0,100]) and output ``ab`` (1x2x256x256, raw ab).
Normalisation is baked into the graph.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from colorizer.core import weights
from colorizer.models.zhang import ECCV16_CHECKPOINT, ECCV16_ONNX, ECCV16_SIZE, onnx_path


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=onnx_path(ECCV16_ONNX))
    args = parser.parse_args()

    ckpt = weights.ensure(ECCV16_CHECKPOINT)
    net = ECCVGenerator()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    net.eval()

    dummy = torch.rand(1, 1, ECCV16_SIZE, ECCV16_SIZE) * 100.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        (dummy,),
        str(args.out),
        input_names=["L"],
        output_names=["ab"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = net(dummy).numpy()
    (got,) = sess.run(None, {"L": dummy.numpy()})
    err = float(np.abs(got - expected).max())
    if err > 1e-2:
        raise SystemExit(f"ONNX output differs from PyTorch (max abs err {err:.4g})")
    logging.info(
        "wrote %s (max abs err vs PyTorch %.2e, sha256 %s)",
        args.out,
        err,
        weights.sha256_file(args.out),
    )


if __name__ == "__main__":
    main()
