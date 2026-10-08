import numpy as np
import pytest

from colorizer.core.restore import (
    RESTORE_PARAMS,
    blend_restored,
    run_restorer,
    validate_restore_params,
)
from colorizer.core.runtime import CPU
from tests.conftest import DummyRestorer


@pytest.fixture
def restorer():
    r = DummyRestorer()
    r.load(CPU)
    return r


def test_shared_params_added_with_defaults():
    assert validate_restore_params(DummyRestorer, {}) == {"scale": 1, "offset": 0.0, "blend": 1.0}
    assert validate_restore_params(DummyRestorer, {"blend": 0.5})["blend"] == 0.5


@pytest.mark.parametrize(
    "bad", [{"scale": 9}, {"blend": 1.5}, {"blend": -0.1}, {"nope": 1}, {"scale": "x"}]
)
def test_bad_params_raise_before_restoring(restorer, bad):
    with pytest.raises(ValueError):
        run_restorer(restorer, np.zeros((4, 4), np.float32), bad)
    assert restorer.calls == []


def test_disabled_restorer_rejected():
    stub = type("Stub", (DummyRestorer,), {"id": "stub", "enabled": False})
    with pytest.raises(ValueError, match="not implemented"):
        validate_restore_params(stub, {})


def test_restore_may_change_resolution_and_is_clipped(restorer):
    gray = np.full((5, 7), 0.8, np.float32)
    out = run_restorer(restorer, gray, {"scale": 2, "offset": 0.5})
    assert out.shape == (10, 14) and out.dtype == np.float32
    assert out.max() == 1.0
    assert restorer.calls == [{"scale": 2, "offset": 0.5}]  # blend is not passed on


def test_blend_mixes_with_bicubic_upscaled_original():
    original = np.random.default_rng(0).uniform(0, 1, (6, 8)).astype(np.float32)
    restored = np.zeros((12, 16), np.float32)
    assert blend_restored(original, restored, 1.0) is restored
    import cv2

    up = cv2.resize(original, (16, 12), interpolation=cv2.INTER_CUBIC)
    assert np.allclose(blend_restored(original, restored, 0.0), np.clip(up, 0, 1))
    assert np.allclose(blend_restored(original, restored, 0.25), np.clip(0.75 * up, 0, 1))


def test_bad_output_shape(restorer, monkeypatch):
    monkeypatch.setattr(restorer, "restore", lambda g, **p: np.zeros((2, 2, 3), np.float32))
    with pytest.raises(ValueError, match="expected float HxW"):
        run_restorer(restorer, np.zeros((2, 2), np.float32), {})


def test_blend_param_schema():
    (blend,) = RESTORE_PARAMS
    assert (blend.name, blend.default, blend.min, blend.max) == ("blend", 1.0, 0.0, 1.0)
