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
from colorizer.core.weights import WeightFile, cache_dir, ensure

# Prebuilt ONNX files, made by tools/export_onnx/* and published as GitHub release assets.
ONNX_RELEASE_URL = "https://github.com/Makistos/colorize/releases/download/models-v1/"
ONNX_SHA256: dict[str, str] = {
    "ddcolor_artistic.onnx": "f426a9cbffa2be0e9d3d7d6925ac513271b6e187296258c73570e9727c8203ad",
    "ddcolor_large.onnx": "73eb1bcf865617354b6b49747d83aff445a1500e1e41a1e9f0183e43ec11507d",
    "ddcolor_tiny.onnx": "bc441aba000b9de6a36e0b71ea35924aed344d92e6865b22fa1e62efbe72d0ce",
    "deoldify_artistic.onnx": "0172f8797adb035098957e86458c5dc0c587b8b295b66f77a3ab4f22dc35b8b3",
    "deoldify_stable.onnx": "37fc9f524f5a6ad13bc4379e3a929829a13fe1015203dcb10f9bfb9a38377cd1",
    "zhang_eccv16.onnx": "ec93605a9e8d792a2853a78524a05d28b66a6876a6eef5600757e589ded01456",
    "zhang_siggraph17.onnx": "5c959374532e87011e3fcd92964be0d6fdb4e29c2c9cdbfe93688e9b9bf5321a",
}


@dataclass(frozen=True)
class OnnxVariant:
    checkpoint: WeightFile  # upstream weights the ONNX file is exported from
    onnx: str  # filename in the cache dir


def onnx_path(filename: str) -> Path:
    return cache_dir() / filename


def ensure_onnx(filename: str, export_hint: str) -> Path:
    """Local path of an ONNX file: a local export in the cache dir, else the hosted build."""
    path = onnx_path(filename)
    if path.exists():
        return path
    sha256 = ONNX_SHA256.get(filename)
    if sha256 is None:
        raise FileNotFoundError(f"{path} not found. Export it once with:\n    {export_hint}")
    return ensure(WeightFile(filename, ONNX_RELEASE_URL + filename, sha256))


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
            path = ensure_onnx(self.variants[variant].onnx, self.export_hint)
            self._sessions[variant], _ = create_session(path, self.device)
        return self._sessions[variant]
