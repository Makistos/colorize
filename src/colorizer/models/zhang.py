"""Zhang et al., "Colorful Image Colorization" (ECCV 2016).

Source: https://github.com/richzhang/colorization (BSD-2-Clause).
The ONNX file is produced by ``tools/export_onnx/zhang.py`` from the official checkpoint
below; until a hosted ONNX build exists, the export must be run once locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

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

EXPORT_HINT = "uv run --group export python tools/export_onnx/zhang.py"


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
