from __future__ import annotations

import os
from typing import Any, ClassVar

import numpy as np
import pytest

from colorizer.core.base import ColorizerModel
from colorizer.core.params import Param
from colorizer.core.registry import Registry
from colorizer.core.restore import Restorer
from colorizer.core.runtime import Device

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class DummyModel(ColorizerModel):
    """Returns constant ab at a small working size. Records calls for assertions."""

    id = "dummy"
    display_name = "Dummy"
    license = "test"
    params = (
        Param("size", "int", 32, min=16, max=128, step=16),
        Param("a", "float", 10.0, min=-100.0, max=100.0),
        Param("b", "float", -20.0, min=-100.0, max=100.0),
    )
    instances: ClassVar[list[DummyModel]] = []

    def __init__(self) -> None:
        self.loaded_on: Device | None = None
        self.unloaded = False
        self.calls: list[dict[str, Any]] = []
        DummyModel.instances.append(self)

    def load(self, device: Device) -> None:
        self.loaded_on = device

    def unload(self) -> None:
        self.unloaded = True

    def predict_ab(self, L: np.ndarray, **params: Any) -> np.ndarray:
        assert L.dtype == np.float32 and L.ndim == 2
        self.calls.append(params)
        ab = np.empty((params["size"], params["size"], 2), np.float32)
        ab[..., 0] = params["a"]
        ab[..., 1] = params["b"]
        return ab


class DummyRestorer(Restorer):
    """Upscales by ``scale`` (nearest) and adds ``offset``. Records calls for assertions."""

    id = "dummy_restore"
    display_name = "Dummy restorer"
    license = "test"
    params = (
        Param("scale", "int", 1, min=1, max=4),
        Param("offset", "float", 0.0, min=-1.0, max=1.0),
    )
    instances: ClassVar[list[DummyRestorer]] = []

    def __init__(self) -> None:
        self.loaded_on: Device | None = None
        self.unloaded = False
        self.calls: list[dict[str, Any]] = []
        DummyRestorer.instances.append(self)

    def load(self, device: Device) -> None:
        self.loaded_on = device

    def unload(self) -> None:
        self.unloaded = True

    def restore(self, gray: np.ndarray, **params: Any) -> np.ndarray:
        assert gray.dtype == np.float32 and gray.ndim == 2
        self.calls.append(params)
        s = params["scale"]
        return np.repeat(np.repeat(gray, s, axis=0), s, axis=1) + params["offset"]


@pytest.fixture
def registry() -> Registry:
    DummyModel.instances.clear()
    DummyRestorer.instances.clear()
    reg = Registry(discover=False)
    reg.register(DummyModel)
    reg.restorers.register(DummyRestorer)
    return reg


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gates = {
        "weights": ("COLORIZER_TEST_WEIGHTS", "downloads/uses model weights"),
        "diffusion": ("COLORIZER_TEST_DIFFUSION", "Stable Diffusion: ~3.6 GB download, slow"),
    }
    for marker, (env, why) in gates.items():
        if os.environ.get(env) == "1":
            continue
        skip = pytest.mark.skip(reason=f"{why}; set {env}=1 to run")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
