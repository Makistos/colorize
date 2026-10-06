from importlib.metadata import EntryPoint

import pytest

from colorizer.core import registry as registry_mod
from colorizer.core.registry import BUILTIN_MODELS, Registry
from colorizer.core.runtime import CPU, Device
from tests.conftest import DummyModel


def test_builtins_discovered_and_importable():
    reg = Registry()
    for model_id in BUILTIN_MODELS:
        assert model_id in reg.ids()
        assert reg.get_class(model_id).id == model_id


def test_entry_point_discovered(monkeypatch):
    ep = EntryPoint("dummy", "tests.conftest:DummyModel", registry_mod.ENTRY_POINT_GROUP)
    monkeypatch.setattr(registry_mod, "entry_points", lambda group: [ep])
    reg = Registry()
    assert reg.get_class("dummy") is DummyModel


def test_duplicate_id_rejected(registry):
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(DummyModel)


def test_unknown_model(registry):
    with pytest.raises(ValueError, match="unknown model"):
        registry.get_class("nope")


def test_mismatched_id_rejected():
    reg = Registry(discover=False)
    reg.register("tests.conftest:DummyModel", "other")
    with pytest.raises(ValueError, match="registered as"):
        reg.get_class("other")


def test_lazy_load_and_cache(registry):
    m1 = registry.get("dummy", CPU)
    assert m1.loaded_on == CPU
    assert registry.get("dummy", CPU) is m1
    assert len(DummyModel.instances) == 1


def test_device_change_reloads(registry):
    m1 = registry.get("dummy", CPU)
    m2 = registry.get("dummy", Device("cuda", "CUDAExecutionProvider"))
    assert m1 is not m2 and m1.unloaded


def test_lru_eviction():
    reg = Registry(max_loaded=2, discover=False)
    for name in ("m1", "m2", "m3"):
        reg.register(type(name, (DummyModel,), {"id": name}))
    a = reg.get("m1", CPU)
    reg.get("m2", CPU)
    reg.get("m1", CPU)  # m1 becomes most recent
    reg.get("m3", CPU)
    assert reg.loaded_ids() == ["m1", "m3"]
    assert not a.unloaded
    reg.unload_all()
    assert reg.loaded_ids() == [] and a.unloaded
