"""Restorers that are registered but not implemented yet (``enabled = False``).

They are listed so the UI and CLI can show them as coming; validation rejects them before
anything runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from colorizer.core.restore import Restorer
from colorizer.core.runtime import Device


class _Stub(Restorer):
    enabled = False
    params = ()

    def load(self, device: Device) -> None:
        raise NotImplementedError(f"{self.id} is not implemented yet")

    def restore(self, gray: np.ndarray, **params: Any) -> np.ndarray:
        raise NotImplementedError(f"{self.id} is not implemented yet")


class SwinIR(_Stub):
    """https://github.com/JingyunLiang/SwinIR (Apache-2.0): transformer SR / denoise."""

    id = "swinir"
    display_name = "SwinIR (not implemented yet)"
    license = "Apache-2.0 (JingyunLiang/SwinIR)"


class SeedVR2(_Stub):
    """https://github.com/ByteDance-Seed/SeedVR (Apache-2.0): one-step diffusion restoration.

    Needs PyTorch and a large diffusion model, like the optional SD extra.
    """

    id = "seedvr2"
    display_name = "SeedVR2 (not implemented yet)"
    license = "Apache-2.0 (ByteDance-Seed/SeedVR)"
