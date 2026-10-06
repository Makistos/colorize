"""The interface every colorization model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from colorizer.core.params import Param, validate_params
from colorizer.core.runtime import Device


class ColorizerModel(ABC):
    id: ClassVar[str]
    display_name: ClassVar[str]
    params: ClassVar[tuple[Param, ...]]
    license: ClassVar[str]

    @abstractmethod
    def load(self, device: Device) -> None: ...

    @abstractmethod
    def predict_ab(self, L: np.ndarray, **params: Any) -> np.ndarray:
        """L: float32 HxW in [0,100]. Returns ab: float32 h x w x 2 (any size; pipeline resizes)."""

    def unload(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release weights and sessions. Called when the registry evicts the model."""

    @classmethod
    def validate_params(cls, values: Mapping[str, Any]) -> dict[str, Any]:
        """Validate ``values`` against ``cls.params``, filling defaults. Raises ``ValueError``."""
        return validate_params(cls.params, values)
