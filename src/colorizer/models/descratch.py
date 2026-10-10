"""Scratch and dust removal: detect damage with a network, then inpaint it.

Detection: the scratch-detection UNet of "Bringing Old Photos Back to Life" (Wan et al.,
CVPR 2020; https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life, MIT), commit
``33875eccf4eb``. Its checkpoint ships only inside the official v1.0
``global_checkpoints.zip``; ``tools/export_onnx/descratch.py`` extracts it and writes the
ONNX file. The network takes grayscale input, like our pipeline.

Upstream detects at a short side of 256 px and thresholds the sigmoid at 0.4; the damaged
pixels are then filled by its large restoration networks. Here they are filled with
OpenCV's Telea inpainting instead, which suits thin scratches and specks.

The network finds light damage reliably (emulsion loss, cracks, white scratches and specks:
100% of thin light lines in tests) but misses dark scratches and dark dust (e.g. dust on the
scanner glass). A classical black-hat detector for those was tried and dropped: it either
missed most dark lines or masked 1-3% of clean photos (hair, thin dark detail).
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

UPSTREAM_COMMIT = "33875eccf4ebcd3665cf38cc56f3a0ce563d3a9c"
CHECKPOINT = WeightFile(
    "bopb_detection_FT_Epoch_latest.pt",
    "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/"
    "global_checkpoints.zip",
    "b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95",
)
CHECKPOINT_ZIP_MEMBER = "checkpoints/detection/FT_Epoch_latest.pt"
ONNX = "scratch_detection.onnx"
EXPORT_HINT = "uv run --group export python tools/export_onnx/descratch.py"
MULTIPLE = 16  # 4 downsamplings
DETECT_SIZES = {"normal": 256, "fine": 512}  # short side used for detection


class Descratch(Restorer):
    id = "descratch"
    display_name = "Scratch & dust removal"
    license = "MIT (microsoft/Bringing-Old-Photos-Back-to-Life detector)"
    params = (
        Param(
            "sensitivity",
            "float",
            0.6,
            min=0.0,
            max=1.0,
            step=0.05,
            help="Higher marks more as damage (0.6 = upstream's threshold).",
        ),
        Param(
            "detail",
            "choice",
            "normal",
            choices=tuple(DETECT_SIZES),
            help="normal: detect at 256 px like upstream. fine: 512 px, finds thinner damage.",
        ),
        Param("grow", "int", 2, min=0, max=10, help="Widen the damage mask by this many pixels."),
    )

    def __init__(self) -> None:
        self.device: Device = CPU
        self._session: Any = None

    def load(self, device: Device) -> None:
        self._session, self.device = create_session(ensure_onnx(ONNX, EXPORT_HINT), device)

    def unload(self) -> None:
        self._session = None

    def damage(self, gray: np.ndarray, detail: str) -> np.ndarray:
        """Probability of damage per pixel (HxW float32), detected at the working size."""
        h, w = gray.shape
        short = DETECT_SIZES[detail]
        scale = short / min(h, w)
        dh = max(MULTIPLE, round(h * scale / MULTIPLE) * MULTIPLE)
        dw = max(MULTIPLE, round(w * scale / MULTIPLE) * MULTIPLE)
        small = cv2.resize(
            gray, (dw, dh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )
        (logits,) = self._session.run(
            None, {"gray": (small * 2 - 1)[None, None].astype(np.float32)}
        )
        prob = 1 / (1 + np.exp(-logits[0, 0]))
        return cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    def restore(self, gray: np.ndarray, **params: Any) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("model not loaded")
        prob = self.damage(gray, params["detail"])
        mask = (prob > 1.0 - params["sensitivity"]).astype(np.uint8)
        grow = int(params["grow"])
        if grow:
            mask = cv2.dilate(
                mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
            )
        if not mask.any():
            return gray.copy()
        img = np.round(np.clip(gray, 0, 1) * 65535).astype(np.uint16)
        radius = 3 + grow
        filled = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
        out: np.ndarray = (filled.astype(np.float32) / 65535).astype(np.float32)
        return out
