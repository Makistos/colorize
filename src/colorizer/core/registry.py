"""Model discovery (built-ins + entry points) and the cache of loaded models."""

from __future__ import annotations

import importlib
import logging
from collections import OrderedDict
from importlib.metadata import entry_points

from colorizer.core.base import ColorizerModel
from colorizer.core.runtime import Device

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "colorizer.models"

# id -> "module:Class". Imported lazily so optional extras don't slow down or break startup.
BUILTIN_MODELS: dict[str, str] = {
    # First entry is the UI's default model.
    "ddcolor": "colorizer.models.ddcolor:DDColor",
    "zhang_eccv16": "colorizer.models.zhang:ZhangECCV16",
    "deoldify": "colorizer.models.deoldify:DeOldify",
}


def _import(target: str) -> type[ColorizerModel]:
    module_name, _, attr = target.partition(":")
    cls = getattr(importlib.import_module(module_name), attr)
    if not (isinstance(cls, type) and issubclass(cls, ColorizerModel)):
        raise TypeError(f"{target} is not a ColorizerModel subclass")
    return cls


class Registry:
    """Known model classes plus an LRU cache of loaded instances."""

    def __init__(self, max_loaded: int = 2, discover: bool = True) -> None:
        if max_loaded < 1:
            raise ValueError("max_loaded must be >= 1")
        self.max_loaded = max_loaded
        self._specs: dict[str, str | type[ColorizerModel]] = {}
        self._loaded: OrderedDict[str, tuple[ColorizerModel, Device]] = OrderedDict()
        if discover:
            for model_id, target in BUILTIN_MODELS.items():
                self.register(target, model_id)
            for ep in entry_points(group=ENTRY_POINT_GROUP):
                self.register(ep.value, ep.name)

    def register(self, model: str | type[ColorizerModel], model_id: str | None = None) -> None:
        """Register a class or lazy ``"module:Class"`` path. Duplicate ids raise ValueError."""
        if model_id is None:
            if isinstance(model, str):
                raise ValueError("model_id is required when registering by import path")
            model_id = model.id
        if model_id in self._specs:
            raise ValueError(f"duplicate model id {model_id!r}")
        self._specs[model_id] = model

    def ids(self) -> list[str]:
        return list(self._specs)

    def get_class(self, model_id: str) -> type[ColorizerModel]:
        try:
            spec = self._specs[model_id]
        except KeyError:
            raise ValueError(f"unknown model {model_id!r}; available: {self.ids()}") from None
        if isinstance(spec, str):
            spec = _import(spec)
            if spec.id != model_id:
                raise ValueError(f"{spec.__name__}.id is {spec.id!r}, registered as {model_id!r}")
            self._specs[model_id] = spec
        return spec

    def get(self, model_id: str, device: Device) -> ColorizerModel:
        """Return a loaded model, loading it on first use and evicting the least recent."""
        cached = self._loaded.get(model_id)
        if cached is not None and cached[1] == device:
            self._loaded.move_to_end(model_id)
            return cached[0]
        if cached is not None:
            self._unload(model_id)
        model = self.get_class(model_id)()
        while len(self._loaded) >= self.max_loaded:
            self._unload(next(iter(self._loaded)))
        log.info("loading %s on %s", model_id, device.name)
        model.load(device)
        self._loaded[model_id] = (model, device)
        return model

    def loaded_ids(self) -> list[str]:
        return list(self._loaded)

    def unload_all(self) -> None:
        for model_id in list(self._loaded):
            self._unload(model_id)

    def _unload(self, model_id: str) -> None:
        model, _ = self._loaded.pop(model_id)
        log.info("unloading %s", model_id)
        model.unload()


_default: Registry | None = None


def default_registry() -> Registry:
    """The process-wide registry (the only global mutable state, by design)."""
    global _default
    if _default is None:
        _default = Registry()
    return _default
