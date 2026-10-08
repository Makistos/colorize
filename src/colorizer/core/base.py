"""The interfaces every plugin implements: colorization models and restorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np

from colorizer.core.params import Param, validate_params
from colorizer.core.runtime import Device


class Plugin(ABC):
    """What models and restorers share: identity, a Param schema, and a load lifecycle."""

    id: ClassVar[str]
    display_name: ClassVar[str]
    params: ClassVar[tuple[Param, ...]]
    license: ClassVar[str]

    # Set by the caller around inference; slow plugins may call it with progress in [0, 1].
    # It raises ``Cancelled`` when the job was cancelled, which should propagate.
    step_callback: Callable[[float], None] | None = None

    @classmethod
    def available(cls) -> bool:
        """False if optional dependencies are missing (the UI then hides the plugin)."""
        return True

    @abstractmethod
    def load(self, device: Device) -> None: ...

    def unload(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release weights and sessions. Called when the registry evicts the plugin."""

    @classmethod
    def validate_params(cls, values: Mapping[str, Any]) -> dict[str, Any]:
        """Validate ``values`` against ``cls.params``, filling defaults. Raises ``ValueError``."""
        return validate_params(cls.params, values)


class ColorizerModel(Plugin):
    @abstractmethod
    def predict_ab(self, L: np.ndarray, **params: Any) -> np.ndarray:
        """L: float32 HxW in [0,100]. Returns ab: float32 h x w x 2 (any size; pipeline resizes)."""
