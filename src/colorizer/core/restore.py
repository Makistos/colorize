"""Optional restoration (denoise, deblur, upscale, faces) that runs before colorization."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import cv2
import numpy as np

from colorizer.core.base import Plugin
from colorizer.core.params import Param, validate_params

# Shared by every restorer, like the postprocess params are shared by every model.
RESTORE_PARAMS: tuple[Param, ...] = (
    Param(
        "blend",
        "float",
        1.0,
        min=0.0,
        max=1.0,
        step=0.05,
        help="Mix with the bicubic-upscaled input (0 = input only); limits invented detail.",
    ),
)
_SHARED_NAMES = frozenset(p.name for p in RESTORE_PARAMS)


@dataclass(frozen=True)
class RestoreStep:
    """One stage of a restore chain: a restorer id and its params (own + shared)."""

    id: str
    params: Mapping[str, Any] = field(default_factory=dict)


class Restorer(Plugin):
    """Restores a grayscale image. Restorers can be chained; each may change resolution."""

    # False for registered stubs that are not implemented yet: listed, but cannot run.
    enabled: ClassVar[bool] = True
    # Shown next to the restorer in the UI, e.g. a non-commercial license.
    warning: ClassVar[str] = ""

    @abstractmethod
    def restore(self, gray: np.ndarray, **params: Any) -> np.ndarray:
        """gray: float32 HxW in [0, 1] (sRGB-encoded). Returns float32 H'xW' in [0, 1]."""


def validate_restore_params(cls: type[Restorer], values: Mapping[str, Any]) -> dict[str, Any]:
    """The restorer's own params plus the shared ones, validated with defaults filled.

    Raises ``ValueError`` for bad params or a disabled restorer.
    """
    if not cls.enabled:
        raise ValueError(f"restorer {cls.id!r} is not implemented yet")
    own = {k: v for k, v in values.items() if k not in _SHARED_NAMES}
    shared = {k: v for k, v in values.items() if k in _SHARED_NAMES}
    return {**cls.validate_params(own), **validate_params(RESTORE_PARAMS, shared)}


def blend_restored(original: np.ndarray, restored: np.ndarray, blend: float) -> np.ndarray:
    """``blend`` * restored + (1 - ``blend``) * original bicubic-resized to the restored size."""
    if blend >= 1.0:
        return restored
    h, w = restored.shape
    if original.shape != (h, w):
        original = cv2.resize(original, (w, h), interpolation=cv2.INTER_CUBIC)
    mixed: np.ndarray = np.clip(blend * restored + (1.0 - blend) * original, 0.0, 1.0)
    return mixed.astype(np.float32)


def run_restorer(restorer: Restorer, gray: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """Validate ``params``, restore ``gray`` and apply the shared ``blend``."""
    validated = validate_restore_params(type(restorer), params)
    blend = validated.pop("blend")
    out = restorer.restore(gray.astype(np.float32, copy=False), **validated)
    if out.ndim != 2 or not np.issubdtype(out.dtype, np.floating):
        raise ValueError(
            f"{restorer.id}.restore returned {out.dtype} {out.shape}, expected float HxW"
        )
    out = np.clip(np.nan_to_num(out.astype(np.float32, copy=False)), 0.0, 1.0)
    return blend_restored(gray, out, blend)
