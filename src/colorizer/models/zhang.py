"""Zhang et al., "Colorful Image Colorization" (ECCV 2016).

Source: https://github.com/richzhang/colorization (BSD-2-Clause).
The ONNX file is produced by ``tools/export_onnx/zhang.py`` from the official checkpoint
below; until a hosted ONNX build exists, the export must be run once locally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.color import rgb2lab

from colorizer.core.base import ColorizerModel
from colorizer.core.params import Param
from colorizer.core.runtime import Device, create_session
from colorizer.core.weights import WeightFile, cache_dir

ECCV16_CHECKPOINT = WeightFile(
    filename="colorization_release_v2-9b330a0b.pth",
    url="https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
    sha256="9b330a0bae53f4ded77b1e23defbf78beaa09c10ebc4c4999e8e4f4a160b93f9",
)
ECCV16_ONNX = "zhang_eccv16.onnx"
ECCV16_SIZE = 256

SIGGRAPH17_CHECKPOINT = WeightFile(
    filename="siggraph17-df00044c.pth",
    url="https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth",
    sha256="df00044c0a4d7c3edcecf6f75437ce346a66e7a42612d9b968e1a7e17dbc6f66",
)
SIGGRAPH17_ONNX = "zhang_siggraph17.onnx"

EXPORT_HINT = "uv run --group export python tools/export_onnx/zhang.py"

log = logging.getLogger(__name__)


def onnx_path(filename: str) -> Path:
    return cache_dir() / filename


class ZhangECCV16(ColorizerModel):
    id = "zhang_eccv16"
    display_name = "Zhang ECCV16"
    license = "BSD-2-Clause (richzhang/colorization)"
    params = (
        Param(
            "size",
            "int",
            ECCV16_SIZE,
            min=ECCV16_SIZE,
            max=ECCV16_SIZE,
            help="Working resolution. Fixed at 256 for this model.",
        ),
    )

    def __init__(self) -> None:
        self._session: Any = None
        self.device: Device | None = None

    def load(self, device: Device) -> None:
        path = onnx_path(ECCV16_ONNX)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Export it once with:\n    {EXPORT_HINT}")
        self._session, self.device = create_session(path, device)

    def unload(self) -> None:
        self._session = None
        self.device = None

    def predict_ab(self, L: np.ndarray, **params: Any) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("model not loaded")
        size = int(params.get("size", ECCV16_SIZE))
        small = cv2.resize(L, (size, size), interpolation=cv2.INTER_AREA)
        (ab,) = self._session.run(None, {"L": small[None, None].astype(np.float32)})
        return np.ascontiguousarray(ab[0].transpose(1, 2, 0), dtype=np.float32)


class ZhangSIGGRAPH17(ColorizerModel):
    """Zhang et al., "Real-Time User-Guided Image Colorization" (SIGGRAPH 2017).

    ``hints`` are user colour points; each sets the colour of its region, and the network
    propagates it along object boundaries.
    """

    id = "zhang_siggraph17"
    display_name = "Zhang SIGGRAPH17 (hints)"
    license = "BSD-2-Clause (richzhang/colorization)"
    params = (
        Param(
            "size",
            "int",
            256,
            min=256,
            max=768,
            step=64,
            help="Working resolution. Trained at 256; higher keeps finer hint placement.",
        ),
        Param(
            "hints",
            "points",
            [],
            help="Colour hints: click the photo to add one with the chosen colour.",
        ),
    )

    def __init__(self) -> None:
        self._session: Any = None
        self.device: Device | None = None

    def load(self, device: Device) -> None:
        path = onnx_path(SIGGRAPH17_ONNX)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Export it once with:\n    {EXPORT_HINT}")
        self._session, self.device = create_session(path, device)

    def unload(self) -> None:
        self._session = None
        self.device = None

    def predict_ab(self, L: np.ndarray, **params: Any) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("model not loaded")
        size = int(params["size"])
        h, w = L.shape
        small = cv2.resize(L, (size, size), interpolation=cv2.INTER_AREA)
        hint_ab, hint_mask = encode_hints(params["hints"], (h, w), size)
        (ab,) = self._session.run(
            None,
            {
                "L": small[None, None].astype(np.float32),
                "hint_ab": hint_ab[None],
                "hint_mask": hint_mask[None],
            },
        )
        return np.ascontiguousarray(ab[0].transpose(1, 2, 0), dtype=np.float32)


def encode_hints(
    hints: list[tuple[int, int, tuple[int, int, int]]], image_hw: tuple[int, int], size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise hint points into (2xSxS ab, 1xSxS mask) at the working size.

    Points are in original-image pixels; each becomes a small square patch (3x3 at 256).
    Points outside the image are ignored.
    """
    h, w = image_hw
    ab = np.zeros((2, size, size), np.float32)
    mask = np.zeros((1, size, size), np.float32)
    r = max(1, size // 256)
    for x, y, rgb in hints:
        if not (0 <= x < w and 0 <= y < h):
            log.warning("ignoring hint at (%d, %d): outside the %dx%d image", x, y, w, h)
            continue
        cx, cy = int(x * size / w), int(y * size / h)
        color = np.array(rgb, np.float64).reshape(1, 1, 3) / 255.0
        a, b = rgb2lab(color)[0, 0, 1:]
        ys, xs = slice(max(cy - r, 0), cy + r + 1), slice(max(cx - r, 0), cx + r + 1)
        ab[0, ys, xs], ab[1, ys, xs] = a, b
        mask[0, ys, xs] = 1.0
    return ab, mask
