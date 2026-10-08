import numpy as np
import pytest
from PIL import Image

from colorizer.core.runtime import CPU
from colorizer.core.session import Session
from tests.conftest import DummyModel


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "img.png"
    Image.fromarray(np.tile(np.arange(0, 200, 4, dtype=np.uint8), (30, 1))).save(p)
    return p


def test_ab_cached_per_image_model_params(registry, image_path):
    session = Session(registry)
    image = session.load(image_path)
    assert session.load(image_path) is image
    ab1 = session.get_ab(image, "dummy", {"size": 32}, CPU)
    ab2 = session.get_ab(image, "dummy", {}, CPU)  # same as explicit defaults
    assert ab2 is ab1 and session.inference_count == 1
    session.get_ab(image, "dummy", {"size": 48}, CPU)
    assert session.inference_count == 2
    assert session.cached_ab(image, "dummy", {"size": 48}) is not None
    assert session.cached_ab(image, "dummy", {"size": 64}) is None


def test_same_pixels_share_cache(registry, image_path, tmp_path):
    copy = tmp_path / "copy.png"
    copy.write_bytes(image_path.read_bytes())
    session = Session(registry)
    session.get_ab(session.load(image_path), "dummy", {}, CPU)
    assert session.cached_ab(session.load(copy), "dummy", {}) is not None


def test_invalid_params_fail_before_loading(registry, image_path):
    session = Session(registry)
    with pytest.raises(ValueError):
        session.get_ab(session.load(image_path), "dummy", {"size": 17}, CPU)
    assert DummyModel.instances == []


def test_restore_cached_across_colorizer_params(registry, image_path):
    from colorizer.core.restore import RestoreStep

    session = Session(registry)
    image = session.load(image_path)
    steps = [RestoreStep("dummy_restore", {"scale": 2})]
    restored = session.get_restored(image, steps, CPU)
    assert restored.src.size == (60, 100) and session.restore_count == 1
    session.get_ab(restored, "dummy", {"size": 32}, CPU)
    # Colorizer params change: restoration is reused, the model runs again.
    again = session.get_restored(image, [RestoreStep("dummy_restore", {"scale": 2})], CPU)
    assert again is restored and session.restore_count == 1
    session.get_ab(again, "dummy", {"size": 48}, CPU)
    assert session.inference_count == 2
    assert session.cached_restored(image, steps) is restored


def test_restore_param_change_invalidates_restore_and_ab(registry, image_path):
    from colorizer.core.restore import RestoreStep

    session = Session(registry)
    image = session.load(image_path)
    a = session.get_restored(image, [RestoreStep("dummy_restore", {"offset": 0.1})], CPU)
    session.get_ab(a, "dummy", {}, CPU)
    b = session.get_restored(image, [RestoreStep("dummy_restore", {"offset": 0.2})], CPU)
    assert b is not a and b.digest != a.digest and session.restore_count == 2
    assert session.cached_ab(b, "dummy", {}) is None
    # Defaults spelled out explicitly hit the same cache entry.
    explicit = RestoreStep("dummy_restore", {"offset": 0.1, "blend": 1.0})
    c = session.get_restored(image, [explicit], CPU)
    assert c is a and session.restore_count == 2


def test_restore_chain_reuses_cached_prefix(registry, image_path):
    from colorizer.core.restore import RestoreStep

    session = Session(registry)
    image = session.load(image_path)
    first = RestoreStep("dummy_restore", {"scale": 2})
    session.get_restored(image, [first, RestoreStep("dummy_restore", {"offset": 0.1})], CPU)
    assert session.restore_count == 2
    session.get_restored(image, [first, RestoreStep("dummy_restore", {"offset": 0.3})], CPU)
    assert session.restore_count == 3  # only the changed last step re-ran
    assert session.get_restored(image, [], CPU) is image


def test_bad_restore_params_fail_before_loading(registry, image_path):
    from colorizer.core.restore import RestoreStep
    from tests.conftest import DummyRestorer

    session = Session(registry)
    image = session.load(image_path)
    steps = [RestoreStep("dummy_restore"), RestoreStep("dummy_restore", {"scale": 99})]
    with pytest.raises(ValueError):
        session.get_restored(image, steps, CPU)
    assert DummyRestorer.instances == []
    with pytest.raises(ValueError, match="unknown restorer"):
        session.get_restored(image, [RestoreStep("nope")], CPU)
