"""Shared plumbing for models that ship one ONNX file per variant."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
from skimage.color import lab2rgb

from colorizer.core.base import ColorizerModel
from colorizer.core.runtime import CPU, Device, create_session
from colorizer.core.weights import WeightFile, cache_dir


@dataclass(frozen=True)
class OnnxVariant:
    checkpoint: WeightFile  # upstream weights the ONNX file is exported from
    onnx: str  # filename in the cache dir


def onnx_path(filename: str) -> Path:
    return cache_dir() / filename


def resize_L(L: np.ndarray, size: int) -> np.ndarray:
    """Square-resize L to the model's working size (the upstream models stretch, not pad)."""
    shrinking = size < max(L.shape)
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(L, (size, size), interpolation=interp)


def gray_rgb(L: np.ndarray) -> np.ndarray:
    """Neutral sRGB image (HxWx3 float32 in [0,1]) with lightness L."""
    lab = np.dstack([L, np.zeros_like(L), np.zeros_like(L)]).astype(np.float64)
    return np.asarray(lab2rgb(lab), dtype=np.float32)


def to_nchw(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)


class OnnxVariantModel(ColorizerModel):
    """Base for models with a ``variant`` param, each variant an ONNX file loaded on demand."""

    variants: ClassVar[Mapping[str, OnnxVariant]]
    export_hint: ClassVar[str]

    def __init__(self) -> None:
        self.device: Device = CPU
        self._sessions: dict[str, Any] = {}

    def load(self, device: Device) -> None:
        self.device = device
        self._sessions.clear()

    def unload(self) -> None:
        self._sessions.clear()

    def session(self, variant: str) -> Any:
        if variant not in self._sessions:
            path = onnx_path(self.variants[variant].onnx)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Export it once with:\n    {self.export_hint}"
                )
            self._sessions[variant], _ = create_session(path, self.device)
        return self._sessions[variant]
