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
