"""Real-ESRGAN super-resolution (realesr-general-x4v3) with adjustable denoising.

Source: https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause), architecture from commit
``a4abfb2979a7``. ONNX files are produced by ``tools/export_onnx/realesrgan.py``.

``denoise_strength`` works as upstream's DNI: the weights are interpolated between the
general model (strong denoise, 1) and its "wdn" twin (weak denoise, 0). The exported graph
holds both weight sets and takes the strength as an input. Scale 2 runs the x4 network and
downsamples, like upstream's ``--outscale``.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from colorizer.core.params import Param
from colorizer.core.restore import Restorer
from colorizer.core.runtime import CPU, Device, create_session
from colorizer.core.weights import WeightFile
from colorizer.models._onnx import ensure_onnx
from colorizer.models._tiles import run_tiled

UPSTREAM_COMMIT = "a4abfb2979a7bbff3f69f58f58ae324608821e27"
_RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
GENERAL_CHECKPOINT = WeightFile(
    "realesr-general-x4v3.pth",
    _RELEASE + "realesr-general-x4v3.pth",
    "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
)
WDN_CHECKPOINT = WeightFile(
    "realesr-general-wdn-x4v3.pth",
    _RELEASE + "realesr-general-wdn-x4v3.pth",
    "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
)
ONNX = "realesrgan_general_x4v3.onnx"
EXPORT_HINT = "uv run --group export python tools/export_onnx/realesrgan.py"
NET_SCALE = 4


class RealESRGAN(Restorer):
    id = "realesrgan"
    display_name = "Real-ESRGAN (upscale)"
    license = "BSD-3-Clause (xinntao/Real-ESRGAN)"
    params = (
        Param("scale", "int", 2, min=2, max=4, step=2, help="Output size factor: 2 or 4."),
        Param(
            "tile",
            "int",
            512,
            min=0,
            max=2048,
            step=32,
            help="Process in tiles of this size to save memory (0 = whole image at once).",
        ),
        Param(
            "denoise_strength",
            "float",
            0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            help="0 keeps grain and noise, 1 removes it more strongly.",
        ),
    )

    def __init__(self) -> None:
        self.device: Device = CPU
        self._session: Any = None

    def load(self, device: Device) -> None:
        self._session, self.device = create_session(ensure_onnx(ONNX, EXPORT_HINT), device)

    def unload(self) -> None:
        self._session = None

    def restore(self, gray: np.ndarray, **params: Any) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("model not loaded")
        denoise = np.array([params["denoise_strength"]], np.float32)
        x = np.ascontiguousarray(np.broadcast_to(gray, (1, 3, *gray.shape)), np.float32)

        def run(piece: np.ndarray) -> np.ndarray:
            (y,) = self._session.run(None, {"image": piece, "denoise": denoise})
            return np.asarray(y)

        y = run_tiled(run, x, NET_SCALE, int(params["tile"]), on_tile=self.step_callback)
        out = np.clip(y[0].mean(axis=0), 0.0, 1.0)
        scale = int(params["scale"])
        if scale != NET_SCALE:
            h, w = gray.shape
            out = cv2.resize(out, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
