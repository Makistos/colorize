"""Export the scratch-detection network of "Bringing Old Photos Back to Life" to ONNX.

    uv run --group export python tools/export_onnx/descratch.py

The UNet (depth 4, 2 convs per level, 64 base filters, bilinear upsampling, anti-aliased
downsampling, batch norm) is re-implemented below from Global/detection_models/networks.py
and antialiasing.py at the pinned commit (MIT, Microsoft), so no upstream code is fetched
or run. The checkpoint ships only inside the 2 GB ``global_checkpoints.zip`` of the
official v1.0 release; it is extracted with HTTP range requests (just that member, about
450 MB with optimizer state), checked against its SHA256, and loaded with ``strict=True``.

Checks: ONNX against PyTorch, and that thin scratches drawn on a real photo are detected
while the photo itself mostly is not.

ONNX graph: input ``gray`` (1x1xHxW in [-1, 1]; H, W multiples of 16, dynamic) -> output
``logits`` (1x1xHxW; sigmoid > 0.4 is a scratch upstream).
"""

from __future__ import annotations

import hashlib
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from colorizer.core import weights
from colorizer.core.runtime import CPU, create_session
from colorizer.models._onnx import onnx_path
from colorizer.models.descratch import CHECKPOINT, CHECKPOINT_ZIP_MEMBER, MULTIPLE, ONNX


class RangeFile(io.RawIOBase):
    """Read-only seekable view of a URL, fetched with HTTP range requests."""

    def __init__(self, url: str) -> None:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.url, self.size = r.url, int(r.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request) as r:
            data = r.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def ensure_checkpoint() -> Path:
    """The detection checkpoint in the cache dir, extracted from the release zip if needed."""
    target = weights.cache_dir() / CHECKPOINT.filename
    if target.exists():
        return target
    logging.info("extracting %s from %s (range requests)", CHECKPOINT_ZIP_MEMBER, CHECKPOINT.url)
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with (
        zipfile.ZipFile(io.BufferedReader(RangeFile(CHECKPOINT.url), 1 << 20)) as archive,
        archive.open(CHECKPOINT_ZIP_MEMBER) as src,
        part.open("wb") as dst,
    ):
        while chunk := src.read(1 << 20):
            digest.update(chunk)
            dst.write(chunk)
    if digest.hexdigest() != CHECKPOINT.sha256:
        part.unlink()
        raise SystemExit(f"checksum mismatch for {CHECKPOINT.filename}: {digest.hexdigest()}")
    part.rename(target)
    return target


class BlurDownsample(nn.Module):
    """Anti-aliased stride-2 downsampling: reflect pad 1, depthwise [1 2 1]^2/16 filter."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        a = torch.tensor([1.0, 2.0, 1.0])
        filt = a[:, None] * a[None, :]
        self.register_buffer("filt", (filt / filt.sum())[None, None].repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filt: torch.Tensor = self.filt
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), filt, stride=2, groups=x.shape[1])


def conv_block(cin: int, cout: int) -> nn.Module:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for c in (cin, cout):
                layers += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(c, cout, 3),
                    nn.BatchNorm2d(cout),
                    nn.LeakyReLU(0.2, True),
                ]
            self.block = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.block(x)
            return out

    return Block()


class UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(cin, cout, 3),
        )
        self.conv_block = conv_block(cin, cout)

    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        return self.conv_block(torch.cat([self.up(x), bridge], 1))


class UNet(nn.Module):
    def __init__(self, depth: int = 4, wf: int = 6) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(1, 2**wf, 7), nn.LeakyReLU(0.2, True)
        )
        prev = 2**wf
        self.down_path, self.down_sample = nn.ModuleList(), nn.ModuleList()
        for i in range(depth):
            self.down_sample.append(
                nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(prev, prev, 3),
                    nn.BatchNorm2d(prev),
                    nn.LeakyReLU(0.2, True),
                    BlurDownsample(prev),
                )
            )
            self.down_path.append(conv_block(prev, 2 ** (wf + i + 1)))
            prev = 2 ** (wf + i + 1)
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(UpBlock(prev, 2 ** (wf + i)))
            prev = 2 ** (wf + i)
        self.last = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(prev, 1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        bridges = []
        for down, block in zip(self.down_sample, self.down_path, strict=True):
            bridges.append(x)
            x = block(down(x))
        for i, up in enumerate(self.up_path):
            x = up(x, bridges[-i - 1])
        out: torch.Tensor = self.last(x)
        return out


def build() -> UNet:
    net = UNet()
    ckpt = torch.load(ensure_checkpoint(), map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model_state"], strict=True)
    return net.eval()


def scratched_photo() -> tuple[np.ndarray, np.ndarray]:
    """scikit-image's "camera" photo (256x256, in [0, 1]) with thin light scratches drawn
    on it, and the mask of where they are."""
    from skimage import data

    img = data.camera()[:256, 128:384].copy()
    mask = np.zeros_like(img)
    for (x0, y0, x1, y1), value in (
        ((10, 20, 240, 200), 245),
        ((200, 10, 60, 250), 235),
        ((30, 150, 250, 120), 15),
    ):
        cv2.line(img, (x0, y0), (x1, y1), value, 2)
        cv2.line(mask, (x0, y0), (x1, y1), 1, 2)
    return img.astype(np.float32) / 255.0, mask.astype(bool)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    net = build()
    out = onnx_path(ONNX)
    out.parent.mkdir(parents=True, exist_ok=True)
    k = MULTIPLE
    h, w = torch.export.Dim("h", min=1, max=512), torch.export.Dim("w", min=1, max=512)
    torch.onnx.export(
        net,
        (torch.rand(1, 1, 256, 320) * 2 - 1,),
        str(out),
        input_names=["gray"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"x": {2: k * h, 3: k * w}},
    )

    session, _ = create_session(out, CPU)
    img, scratches = scratched_photo()
    err = 0.0
    for crop in (img, img[:128, :192]):
        x = torch.from_numpy(crop[None, None] * 2 - 1)
        with torch.no_grad():
            expected = net(x).numpy()
        (got,) = session.run(None, {"gray": x.numpy()})
        err = max(err, float(np.abs(got - expected).max()))
    if err > 1e-3:
        raise SystemExit(f"ONNX differs from PyTorch (max abs err {err:.4g})")
    (logits,) = session.run(None, {"gray": img[None, None] * 2 - 1})
    detected = 1 / (1 + np.exp(-logits[0, 0])) > 0.4
    recall = float(detected[scratches].mean())
    false_alarm = float(
        detected[~cv2.dilate(scratches.astype(np.uint8), np.ones((7, 7))).astype(bool)].mean()
    )
    if recall < 0.5 or false_alarm > 0.05:
        raise SystemExit(
            f"detection check failed: recall {recall:.2f}, false alarm {false_alarm:.3f}"
        )
    logging.info(
        "wrote %s (max abs err %.2e; scratch recall %.2f, false alarm %.3f)",
        out, err, recall, false_alarm,
    )  # fmt: skip


if __name__ == "__main__":
    main()
