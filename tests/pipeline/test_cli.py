import json

import numpy as np
import pytest
from PIL import Image

from colorizer import cli
from tests.conftest import DummyModel


@pytest.fixture
def images(tmp_path):
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    gray = np.tile(np.arange(0, 250, 5, dtype=np.uint8), (20, 1))
    Image.fromarray(gray).save(src / "a.jpg")
    Image.fromarray(gray).save(src / "b.png")
    Image.fromarray(gray).save(src / "sub" / "c.tif")
    (src / "notes.txt").write_text("not an image")
    return src


def run(registry, *args):
    return cli.main([*args, "--model", "dummy", "--device", "cpu"], registry=registry)


def test_single_file_to_dir(registry, images, tmp_path):
    assert run(registry, str(images / "a.jpg"), str(tmp_path / "out")) == 0
    assert (tmp_path / "out" / "a.jpg").exists()


def test_single_file_to_file_infers_format(registry, images, tmp_path):
    assert run(registry, str(images / "a.jpg"), str(tmp_path / "x.png")) == 0
    assert Image.open(tmp_path / "x.png").format == "PNG"


def test_directory_non_recursive_and_recursive(registry, images, tmp_path):
    out = tmp_path / "out"
    assert run(registry, str(images), str(out), "--format", "png") == 0
    assert sorted(p.name for p in out.rglob("*.png")) == ["a.png", "b.png"]
    assert run(registry, str(images), str(out), "-r", "--format", "png") == 0
    assert (out / "sub" / "c.png").exists()


def test_skip_existing_unless_overwrite(registry, images, tmp_path, caplog):
    out = tmp_path / "out"
    run(registry, str(images / "a.jpg"), str(out))
    calls = len(DummyModel.instances[0].calls)
    run(registry, str(images / "a.jpg"), str(out))
    assert len(DummyModel.instances[0].calls) == calls
    assert "1 skipped" in caplog.text
    run(registry, str(images / "a.jpg"), str(out), "--overwrite")
    assert len(DummyModel.instances[0].calls) == calls + 1


def test_params_routed_to_model_and_postprocess(registry, images, tmp_path):
    assert (
        run(
            registry,
            str(images / "a.jpg"),
            str(tmp_path / "o.png"),
            "--param",
            "size=64",
            "--param",
            "a=0",
            "--param",
            "b=0",
            "--param",
            "temperature=1",
        )
        == 0
    )
    assert DummyModel.instances[0].calls[0]["size"] == 64
    out = np.asarray(Image.open(tmp_path / "o.png")).astype(int)
    assert (out[..., 0] - out[..., 2]).mean() > 5  # warmed by temperature only


@pytest.mark.parametrize("bad", ["size=17", "nope=1", "size", "saturation=9"])
def test_bad_params_exit_2_without_loading(registry, images, tmp_path, bad):
    assert run(registry, str(images / "a.jpg"), str(tmp_path / "o"), "--param", bad) == 2
    assert DummyModel.instances == []


def test_preset_then_cli_override(registry, images, tmp_path):
    preset = tmp_path / "p.json"
    preset.write_text(
        json.dumps(
            {"model": "dummy", "params": {"size": 48, "a": 5}, "postprocess": {"saturation": 0.5}}
        )
    )
    assert (
        run(
            registry,
            str(images / "a.jpg"),
            str(tmp_path / "o.png"),
            "--preset",
            str(preset),
            "--param",
            "a=7",
        )
        == 0
    )
    assert DummyModel.instances[0].calls[0] == {"size": 48, "a": 7.0, "b": -20.0}


def test_failure_gives_nonzero_exit_and_summary(registry, images, tmp_path, caplog):
    (images / "broken.png").write_bytes(b"not a png")
    assert run(registry, str(images), str(tmp_path / "out")) == 1
    assert "2 colorized, 0 skipped, 1 failed" in caplog.text
    assert "broken.png" in caplog.text


def test_missing_input(registry, tmp_path):
    assert run(registry, str(tmp_path / "missing.jpg"), str(tmp_path / "o")) == 2


def test_list_models(caplog):
    assert cli.main(["--list-models"]) == 0
    assert "zhang_eccv16" in caplog.text and "saturation" in caplog.text
