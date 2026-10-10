"""SCUNet blind real-image denoising (Zhang et al., Machine Intelligence Research 2023).

Source: https://github.com/cszn/SCUNet (Apache-2.0), architecture from commit
``52e440a80a65``. The "real" models are trained on a synthetic pipeline of mixed noise, blur
and JPEG meant for real photos, which suits film grain and scan noise. Weights: the official
KAIR v1.0 release files. ONNX files are produced by ``tools/export_onnx/scunet.py``.

Gray input is replicated to RGB and the output averaged back. Inference runs in 256 px
tiles with 32 px of context, every tile edge-padded (as upstream pads) to the same 320 px
square: memory grows with tile area (about 11 KB per pixel in ONNX Runtime) and stays flat
only when every run has the same input shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from colorizer.core.params import Param
from colorizer.core.restore import Restorer
from colorizer.core.runtime import CPU, Device, create_session
from colorizer.core.weights import WeightFile
from colorizer.models._onnx import ensure_onnx
from colorizer.models._tiles import run_tiled

UPSTREAM_COMMIT = "52e440a80a655b01e0b41e9dd9bfe599bc11625e"
_RELEASE = "https://github.com/cszn/KAIR/releases/download/v1.0/"
CHECKPOINTS = {
    "psnr": WeightFile(
        "scunet_color_real_psnr.pth",
        _RELEASE + "scunet_color_real_psnr.pth",
        "fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e",
    ),
    "gan": WeightFile(
        "scunet_color_real_gan.pth",
        _RELEASE + "scunet_color_real_gan.pth",
        "892c83f812c59173273b74f4f34a14ecaf57a2fdb68df056664589beb55c966e",
    ),
}
ONNX = {"psnr": "scunet_real_psnr.onnx", "gan": "scunet_real_gan.onnx"}
EXPORT_HINT = "uv run --group export python tools/export_onnx/scunet.py"
TILE, TILE_PAD = 256, 32
MULTIPLE = 64  # 3 downsamplings x window size 8
RUN_SIZE = TILE + 2 * TILE_PAD  # 320, a multiple of 64: one input shape for every tile


class SCUNet(Restorer):
    id = "scunet"
    display_name = "SCUNet (denoise grain)"
    license = "Apache-2.0 (cszn/SCUNet)"
    params = (
        Param(
            "variant",
            "choice",
            "psnr",
            choices=tuple(ONNX),
            help="psnr: smooth, faithful. gan: keeps more texture, may invent some.",
        ),
    )

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
            path = ensure_onnx(ONNX[variant], EXPORT_HINT)
            self._sessions[variant], _ = create_session(path, self.device)
        return self._sessions[variant]

    def restore(self, gray: np.ndarray, **params: Any) -> np.ndarray:
        session = self.session(params["variant"])
        x = np.ascontiguousarray(np.broadcast_to(gray, (1, 3, *gray.shape)), np.float32)

        def run(piece: np.ndarray) -> np.ndarray:
            _, _, h, w = piece.shape
            ph, pw = max(RUN_SIZE - h, -h % MULTIPLE), max(RUN_SIZE - w, -w % MULTIPLE)
            padded = np.pad(piece, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="edge")
            (y,) = session.run(None, {"image": padded})
            return np.asarray(y)[:, :, :h, :w]

        y = run_tiled(run, x, 1, TILE, TILE_PAD, on_tile=self.step_callback)
        out: np.ndarray = np.clip(y[0].mean(axis=0), 0.0, 1.0).astype(np.float32)
        return out
