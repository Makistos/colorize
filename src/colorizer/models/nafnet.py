"""NAFNet image restoration (Chen et al., ECCV 2022): deblurring and denoising.

Source: https://github.com/megvii-research/NAFNet (MIT), architecture from commit
``2b4af71ebe09``. Official weights (NAFNet-GoPro-width64 for deblur, NAFNet-SIDD-width64 for
denoise) are linked from the upstream README on Google Drive; they are downloaded from a
Hugging Face mirror pinned by commit, whose files are byte-identical to the Google Drive
originals (same SHA256). ONNX files are produced by ``tools/export_onnx/nafnet.py``.

Inference runs in 384 px tiles. Besides bounding memory, this matches how upstream
evaluates the GoPro model (NAFNetLocal pools over 384 px windows, 1.5x the training crops).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from colorizer.core.params import Param
from colorizer.core.restore import Restorer
from colorizer.core.runtime import CPU, Device, create_session
from colorizer.core.weights import WeightFile
from colorizer.models._onnx import ensure_onnx
from colorizer.models._tiles import run_tiled

UPSTREAM_REPO = "https://github.com/megvii-research/NAFNet.git"
UPSTREAM_COMMIT = "2b4af71ebe098a92a75910c233a3965a3e93ede4"
_MIRROR = (
    "https://huggingface.co/mikestealth/nafnet-models/resolve/"
    "cd3a91ddc41c9c3a53ee809b3ddd5263c9e94e51/"
)
EXPORT_HINT = "uv run --group export python tools/export_onnx/nafnet.py"
TILE, TILE_PAD = 384, 32
MULTIPLE = 16  # the network downsamples 4 times


@dataclass(frozen=True)
class NAFNetVariant:
    checkpoint: WeightFile
    onnx: str
    # Upstream network config (options/test/<dataset>/NAFNet-width64.yml).
    enc_blk_nums: tuple[int, ...]
    middle_blk_num: int
    dec_blk_nums: tuple[int, ...]


VARIANTS: dict[str, NAFNetVariant] = {
    "deblur": NAFNetVariant(
        WeightFile(
            "NAFNet-GoPro-width64.pth",
            _MIRROR + "NAFNet-GoPro-width64.pth",
            "329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f",
        ),
        "nafnet_deblur.onnx",
        (1, 1, 1, 28),
        1,
        (1, 1, 1, 1),
    ),
    "denoise": NAFNetVariant(
        WeightFile(
            "NAFNet-SIDD-width64.pth",
            _MIRROR + "NAFNet-SIDD-width64.pth",
            "cd685efaae01f7c4e9951f2deab05780079c8eb1e49ed664b72f6db04dabb445",
        ),
        "nafnet_denoise.onnx",
        (2, 2, 4, 8),
        12,
        (2, 2, 2, 2),
    ),
}


class NAFNet(Restorer):
    id = "nafnet"
    display_name = "NAFNet (deblur / denoise)"
    license = "MIT (megvii-research/NAFNet)"
    params = (
        Param(
            "variant",
            "choice",
            "denoise",
            choices=tuple(VARIANTS),
            help="deblur: motion/camera blur (GoPro). denoise: sensor noise and grain (SIDD).",
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
            path = ensure_onnx(VARIANTS[variant].onnx, EXPORT_HINT)
            self._sessions[variant], _ = create_session(path, self.device)
        return self._sessions[variant]

    def restore(self, gray: np.ndarray, **params: Any) -> np.ndarray:
        session = self.session(params["variant"])
        x = np.ascontiguousarray(np.broadcast_to(gray, (1, 3, *gray.shape)), np.float32)

        def run(piece: np.ndarray) -> np.ndarray:
            _, _, h, w = piece.shape
            ph, pw = -h % MULTIPLE, -w % MULTIPLE
            padded = np.pad(piece, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
            (y,) = session.run(None, {"image": padded})
            return np.asarray(y)[:, :, :h, :w]

        y = run_tiled(run, x, 1, TILE, TILE_PAD, on_tile=self.step_callback)
        out: np.ndarray = np.clip(y[0].mean(axis=0), 0.0, 1.0).astype(np.float32)
        return out
