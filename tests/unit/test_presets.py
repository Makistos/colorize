import json

import pytest

from colorizer.core import presets
from colorizer.core.presets import Preset


def test_save_list_load_delete(tmp_path, registry):
    p = Preset("dummy", {"size": 48}, {"saturation": 1.5}).validate(registry)
    assert p.params == {"size": 48, "a": 10.0, "b": -20.0}  # defaults filled
    assert p.postprocess["saturation"] == 1.5 and p.postprocess["tint"] == 0.0
    presets.save_preset("Warm portraits", p, tmp_path)
    presets.save_preset("archive", p, tmp_path)
    assert presets.list_presets(tmp_path) == ["archive", "Warm portraits"]
    assert presets.load_preset("Warm portraits", tmp_path) == p
    presets.delete_preset("archive", tmp_path)
    assert presets.list_presets(tmp_path) == ["Warm portraits"]


def test_load_from_file_path(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"model": "dummy", "params": {"a": 1}}))
    assert presets.load_preset(str(f)) == Preset("dummy", {"a": 1}, {})


@pytest.mark.parametrize("name", ["", "../evil", "a/b", ".hidden", "x" * 100])
def test_bad_names_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        presets.save_preset(name, Preset("dummy"), tmp_path)


@pytest.mark.parametrize(
    "data", [[], {"params": {}}, {"model": "dummy", "params": []}, {"model": 3}]
)
def test_bad_structure(data):
    with pytest.raises(ValueError):
        Preset.from_mapping(data)


def test_bad_values_and_missing(tmp_path, registry):
    with pytest.raises(ValueError):
        Preset("dummy", {"size": 17}).validate(registry)
    with pytest.raises(ValueError):
        Preset("dummy", {}, {"saturation": 9}).validate(registry)
    with pytest.raises(ValueError, match="not found"):
        presets.load_preset("nope", tmp_path)


def test_config_dir_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert presets.presets_dir() == tmp_path / "colorizer" / "presets"


def test_restore_chain_round_trip_and_validation(registry, tmp_path):
    from colorizer.core.presets import Preset, load_preset, save_preset
    from colorizer.core.restore import RestoreStep

    preset = Preset("dummy", restore=(RestoreStep("dummy_restore", {"scale": 2}),))
    save_preset("r", preset, tmp_path)
    loaded = load_preset("r", tmp_path)
    assert loaded.restore == (RestoreStep("dummy_restore", {"scale": 2}),)
    validated = loaded.validate(registry)
    assert validated.restore[0].params == {"scale": 2, "offset": 0.0, "blend": 1.0}
    assert load_preset("r", tmp_path).to_json() == preset.to_json()
    assert '"restore"' not in Preset("dummy").to_json()  # older presets stay unchanged
    for bad in ({"model": "dummy", "restore": {}}, {"model": "dummy", "restore": [{"x": 1}]}):
        with pytest.raises(ValueError):
            Preset.from_mapping(bad)
    with pytest.raises(ValueError):
        Preset("dummy", restore=(RestoreStep("dummy_restore", {"scale": 9}),)).validate(registry)
