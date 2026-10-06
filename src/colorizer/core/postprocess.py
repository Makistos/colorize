"""Model-independent adjustments applied to the ab channels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from colorizer.core.params import Param, validate_params

# ab units added per unit of temperature/tint. ±1 gives a clearly visible but not absurd cast.
SHIFT_SCALE = 30.0

POSTPROCESS_PARAMS: tuple[Param, ...] = (
    Param(
        "saturation",
        "float",
        1.0,
        min=0.0,
        max=2.0,
        step=0.05,
        help="Scale colour intensity (0 = grayscale).",
    ),
    Param(
        "temperature",
        "float",
        0.0,
        min=-1.0,
        max=1.0,
        step=0.05,
        help="Shift towards blue (-) or yellow (+).",
    ),
    Param(
        "tint",
        "float",
        0.0,
        min=-1.0,
        max=1.0,
        step=0.05,
        help="Shift towards green (-) or magenta (+).",
    ),
    Param(
        "chroma_blend",
        "float",
        1.0,
        min=0.0,
        max=1.0,
        step=0.05,
        help="Blend between grayscale (0) and full colour (1).",
    ),
    Param(
        "ab_smooth",
        "float",
        0.0,
        min=0.0,
        max=5.0,
        step=0.1,
        help="Gaussian blur sigma (pixels) on colour only; reduces blotches.",
    ),
)


@dataclass(frozen=True)
class Postprocess:
    saturation: float = 1.0
    temperature: float = 0.0
    tint: float = 0.0
    chroma_blend: float = 1.0
    ab_smooth: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Postprocess:
        """Validate and build from a dict; raises ``ValueError`` on bad input."""
        return cls(**validate_params(POSTPROCESS_PARAMS, values))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def apply(ab: np.ndarray, post: Postprocess) -> np.ndarray:
    """Return adjusted ab (HxWx2 float32). Order: smooth, saturate, shift, blend."""
    out: np.ndarray = ab.astype(np.float32, copy=True)
    if post.ab_smooth > 0:
        out = cv2.GaussianBlur(
            out, (0, 0), sigmaX=post.ab_smooth, sigmaY=post.ab_smooth, borderType=cv2.BORDER_REFLECT
        )
    out *= post.saturation
    out[..., 0] += post.tint * SHIFT_SCALE
    out[..., 1] += post.temperature * SHIFT_SCALE
    out *= post.chroma_blend
    return out
