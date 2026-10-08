"""CodeFormer face restoration (Zhou et al., NeurIPS 2022).

Source: https://github.com/sczhou/CodeFormer, architecture from commit ``b33cc7d639d6``.
License: S-Lab License 1.0, NON-COMMERCIAL use only. Weights: the official v0.1.0 release.
The ONNX file is produced by ``tools/export_onnx/codeformer.py``.

Faces are found with OpenCV's YuNet detector (https://github.com/opencv/opencv_zoo, MIT;
model from the opencv/face_detection_yunet Hugging Face repo pinned by commit), aligned to
the FFHQ 5-point template, restored at 512x512 and pasted back with a feathered mask, as
upstream's FaceRestoreHelper does without its face-parsing network. ``upscale_bg`` doubles
the image with Real-ESRGAN (realesr-general-x4v3; upstream uses RealESRGAN_x2plus).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from colorizer.core.params import Param
from colorizer.core.restore import Restorer
from colorizer.core.runtime import CPU, Device, create_session
from colorizer.core.weights import WeightFile, ensure
from colorizer.models._onnx import ensure_onnx
from colorizer.models.realesrgan import RealESRGAN

UPSTREAM_COMMIT = "b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b"
CHECKPOINT = WeightFile(
    "codeformer.pth",
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
)
DETECTOR = WeightFile(
    "face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/"
    "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
ONNX = "codeformer.onnx"
EXPORT_HINT = "uv run --group export python tools/export_onnx/codeformer.py"
FACE_SIZE = 512
# FFHQ 5-point template at 512x512 (facexlib): eyes, nose tip, mouth corners, ordered
# left to right as seen in the image.
FACE_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    np.float32,
)
DETECT_SCORE = 0.7
MIN_FACE = 16  # px; smaller detections are ignored
BG_SCALE = 2


def detect_faces(detector: Any, gray: np.ndarray) -> list[np.ndarray]:
    """5 landmarks (5x2, image order as in FACE_TEMPLATE) per face found in ``gray``."""
    img = cv2.cvtColor(np.round(gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    h, w = gray.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    found = []
    for f in faces if faces is not None else []:
        if min(f[2], f[3]) < MIN_FACE:
            continue
        # YuNet: box (4), then right eye, left eye, nose tip, right and left mouth corner
        # (the subject's right and left), then score.
        found.append(np.asarray(f[4:14], np.float32).reshape(5, 2))
    return found


def paste_face(canvas: np.ndarray, face: np.ndarray, inverse: np.ndarray, scale: int) -> np.ndarray:
    """Blend an aligned ``face`` into ``canvas`` with a feathered mask (upstream's maths)."""
    h, w = canvas.shape
    inverse = inverse * scale
    if scale > 1:
        inverse[:, 2] += 0.5 * scale  # upstream's offset for precise back-alignment
    restored = cv2.warpAffine(face, inverse, (w, h))
    mask = cv2.warpAffine(np.ones((FACE_SIZE, FACE_SIZE), np.float32), inverse, (w, h))
    mask = cv2.erode(mask, np.ones((2 * scale, 2 * scale), np.uint8))
    pasted = mask * restored
    w_edge = int(np.sum(mask) ** 0.5) // 20
    if w_edge > 0:
        mask = cv2.erode(mask, np.ones((2 * w_edge, 2 * w_edge), np.uint8))
        mask = cv2.GaussianBlur(mask, (2 * w_edge + 1, 2 * w_edge + 1), 0)
    out: np.ndarray = mask * pasted + (1 - mask) * canvas
    return out


class CodeFormer(Restorer):
    id = "codeformer"
    display_name = "CodeFormer (faces)"
    license = "S-Lab License 1.0, non-commercial use only (sczhou/CodeFormer)"
    warning = "CodeFormer is licensed for non-commercial use only (S-Lab License 1.0)."
    params = (
        Param(
            "fidelity",
            "float",
            0.7,
            min=0.0,
            max=1.0,
            step=0.05,
            help="0 = cleaner but more invented faces; 1 = closer to the original face.",
        ),
        Param(
            "upscale_bg",
            "bool",
            False,
            help="Also upscale the whole image 2x with Real-ESRGAN (faces are pasted onto it).",
        ),
    )

    def __init__(self) -> None:
        self.device: Device = CPU
        self._session: Any = None
        self._detector: Any = None
        self._bg: RealESRGAN | None = None

    def load(self, device: Device) -> None:
        self._session, self.device = create_session(ensure_onnx(ONNX, EXPORT_HINT), device)
        self._detector = cv2.FaceDetectorYN.create(str(ensure(DETECTOR)), "", (320, 320))
        self._detector.setScoreThreshold(DETECT_SCORE)
        self._bg = None

    def unload(self) -> None:
        self._session = self._detector = self._bg = None

    def background(self, gray: np.ndarray) -> np.ndarray:
        if self._bg is None:
            bg = RealESRGAN()
            bg.load(self.device)
            self._bg = bg
        params = {p.name: p.default for p in RealESRGAN.params} | {"scale": BG_SCALE}
        return self._bg.restore(gray, **params)

    def restore_face(self, face: np.ndarray, fidelity: float) -> np.ndarray:
        """Aligned gray face (512x512 in [0,1]) -> restored gray face, tone-matched."""
        x = np.ascontiguousarray(np.broadcast_to(face * 2 - 1, (1, 3, FACE_SIZE, FACE_SIZE)))
        feeds = {"face": x.astype(np.float32), "fidelity": np.array([fidelity], np.float32)}
        (y,) = self._session.run(None, feeds)
        out = (np.clip(y[0].mean(axis=0), -1, 1) + 1) / 2
        # Upstream, for gray input: match the restored face's mean/std to the input face.
        std = out.std()
        if std > 1e-6:
            out = (out - out.mean()) / std * face.std() + face.mean()
        result: np.ndarray = np.clip(out, 0, 1)
        return result

    def restore(self, gray: np.ndarray, **params: Any) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("model not loaded")
        scale = BG_SCALE if params["upscale_bg"] else 1
        canvas = (self.background(gray) if scale > 1 else gray).astype(np.float64)
        faces = detect_faces(self._detector, gray)
        for n, landmarks in enumerate(faces):
            affine = cv2.estimateAffinePartial2D(landmarks, FACE_TEMPLATE, method=cv2.LMEDS)[0]
            if affine is None:
                continue
            aligned = cv2.warpAffine(
                gray,
                affine,
                (FACE_SIZE, FACE_SIZE),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=133 / 255,  # upstream's gray fill
            )
            restored = self.restore_face(aligned, float(params["fidelity"]))
            inverse = cv2.invertAffineTransform(affine)
            canvas = paste_face(canvas, restored, inverse, scale)
            if self.step_callback:
                self.step_callback((n + 1) / len(faces))
        out: np.ndarray = np.clip(canvas, 0, 1).astype(np.float32)
        return out
