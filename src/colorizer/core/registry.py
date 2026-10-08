"""Plugin discovery (built-ins + entry points) and the caches of loaded models/restorers."""

from __future__ import annotations

import importlib
import logging
from collections import OrderedDict
from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Generic, TypeVar, cast

from colorizer.core.base import ColorizerModel, Plugin
from colorizer.core.restore import Restorer
from colorizer.core.runtime import Device

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "colorizer.models"
RESTORER_ENTRY_POINT_GROUP = "colorizer.restorers"

# id -> "module:Class". Imported lazily so optional extras don't slow down or break startup.
BUILTIN_MODELS: dict[str, str] = {
    # First entry is the UI's default model.
    "ddcolor": "colorizer.models.ddcolor:DDColor",
    "zhang_eccv16": "colorizer.models.zhang:ZhangECCV16",
    "zhang_siggraph17": "colorizer.models.zhang:ZhangSIGGRAPH17",
    "deoldify": "colorizer.models.deoldify:DeOldify",
    "sd_controlnet": "colorizer.models.sd_controlnet:SDControlNet",
}
BUILTIN_RESTORERS: dict[str, str] = {}

P = TypeVar("P", bound=Plugin)


class PluginRegistry(Generic[P]):
    """Known plugin classes of one kind plus an LRU cache of loaded instances."""

    def __init__(
        self,
        base: type,  # P itself; abstract, so mypy won't accept it as type[P]
        kind: str,
        builtins: Mapping[str, str],
        group: str,
        max_loaded: int = 2,
        discover: bool = True,
    ) -> None:
        if max_loaded < 1:
            raise ValueError("max_loaded must be >= 1")
        self.base, self.kind = base, kind
        self.max_loaded = max_loaded
        self._specs: dict[str, str | type[P]] = {}
        self._loaded: OrderedDict[str, tuple[P, Device]] = OrderedDict()
        if discover:
            for plugin_id, target in builtins.items():
                self.register(target, plugin_id)
            for ep in entry_points(group=group):
                self.register(ep.value, ep.name)

    def register(self, plugin: str | type[P], plugin_id: str | None = None) -> None:
        """Register a class or lazy ``"module:Class"`` path. Duplicate ids raise ValueError."""
        if plugin_id is None:
            if isinstance(plugin, str):
                raise ValueError(f"{self.kind}_id is required when registering by import path")
            plugin_id = plugin.id
        if plugin_id in self._specs:
            raise ValueError(f"duplicate {self.kind} id {plugin_id!r}")
        self._specs[plugin_id] = plugin

    def ids(self) -> list[str]:
        return list(self._specs)

    def get_class(self, plugin_id: str) -> type[P]:
        try:
            spec = self._specs[plugin_id]
        except KeyError:
            raise ValueError(
                f"unknown {self.kind} {plugin_id!r}; available: {self.ids()}"
            ) from None
        if isinstance(spec, str):
            spec = self._import(spec)
            if spec.id != plugin_id:
                raise ValueError(f"{spec.__name__}.id is {spec.id!r}, registered as {plugin_id!r}")
            self._specs[plugin_id] = spec
        return spec

    def _import(self, target: str) -> type[P]:
        module_name, _, attr = target.partition(":")
        cls = getattr(importlib.import_module(module_name), attr)
        if not (isinstance(cls, type) and issubclass(cls, self.base)):
            raise TypeError(f"{target} is not a {self.base.__name__} subclass")
        return cast("type[P]", cls)

    def get(self, plugin_id: str, device: Device) -> P:
        """Return a loaded plugin, loading it on first use and evicting the least recent."""
        cached = self._loaded.get(plugin_id)
        if cached is not None and cached[1] == device:
            self._loaded.move_to_end(plugin_id)
            return cached[0]
        if cached is not None:
            self._unload(plugin_id)
        plugin = self.get_class(plugin_id)()
        while len(self._loaded) >= self.max_loaded:
            self._unload(next(iter(self._loaded)))
        log.info("loading %s on %s", plugin_id, device.name)
        plugin.load(device)
        self._loaded[plugin_id] = (plugin, device)
        return plugin

    def loaded_ids(self) -> list[str]:
        return list(self._loaded)

    def unload_all(self) -> None:
        for plugin_id in list(self._loaded):
            self._unload(plugin_id)

    def _unload(self, plugin_id: str) -> None:
        plugin, _ = self._loaded.pop(plugin_id)
        log.info("unloading %s", plugin_id)
        plugin.unload()


class Registry(PluginRegistry[ColorizerModel]):
    """Colorization models; restorers live in ``self.restorers`` with their own cache."""

    def __init__(self, max_loaded: int = 2, discover: bool = True) -> None:
        super().__init__(
            ColorizerModel, "model", BUILTIN_MODELS, ENTRY_POINT_GROUP, max_loaded, discover
        )
        self.restorers: PluginRegistry[Restorer] = PluginRegistry(
            Restorer,
            "restorer",
            BUILTIN_RESTORERS,
            RESTORER_ENTRY_POINT_GROUP,
            max_loaded,
            discover,
        )

    def unload_all(self) -> None:
        super().unload_all()
        self.restorers.unload_all()


_default: Registry | None = None


def default_registry() -> Registry:
    """The process-wide registry (the only global mutable state, by design)."""
    global _default
    if _default is None:
        _default = Registry()
    return _default
