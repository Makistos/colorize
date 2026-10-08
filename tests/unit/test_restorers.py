import numpy as np
import pytest

from colorizer.core import weights
from colorizer.core.restore import run_restorer
from colorizer.core.runtime import CPU
from colorizer.models import _onnx, realesrgan
from colorizer.models._tiles import run_tiled
from colorizer.models.realesrgan import RealESRGAN


def nearest_x4(x):
    """Stand-in network: position-dependent, so misplaced tiles would show."""
    return np.repeat(np.repeat(x * 0.5 + 0.1, 4, axis=2), 4, axis=3)


@pytest.mark.parametrize("tile", [0, 7, 16, 64])
def test_tiled_equals_whole(tile):
    x = np.random.default_rng(0).uniform(0, 1, (1, 3, 23, 41)).astype(np.float32)
    progress = []
    out = run_tiled(nearest_x4, x, 4, tile, pad=3, on_tile=progress.append)
    assert np.array_equal(out, nearest_x4(x))
    if 0 < tile < 41:
        assert progress[-1] == 1.0 and len(progress) > 1


class FakeSession:
    def __init__(self):
        self.feeds = []

    def run(self, _, feeds):
        self.feeds.append(feeds)
        return [nearest_x4(feeds["image"])]


@pytest.fixture
def esrgan(monkeypatch, tmp_path):
    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    (tmp_path / realesrgan.ONNX).touch()
    session = FakeSession()
    monkeypatch.setattr(realesrgan, "create_session", lambda path, device: (session, device))
    model = RealESRGAN()
    model.load(CPU)
    return model, session


@pytest.mark.parametrize("scale", [2, 4])
def test_realesrgan_scale_and_params(esrgan, scale):
    model, session = esrgan
    gray = np.random.default_rng(1).uniform(0, 1, (30, 50)).astype(np.float32)
    out = run_restorer(model, gray, {"scale": scale, "tile": 32, "denoise_strength": 0.25})
    assert out.shape == (30 * scale, 50 * scale) and out.dtype == np.float32
    assert all(f["denoise"].tolist() == [0.25] for f in session.feeds)
    assert all(f["image"].shape[1] == 3 for f in session.feeds)


@pytest.mark.parametrize("bad", [{"scale": 3}, {"tile": 33}, {"denoise_strength": 1.5}])
def test_realesrgan_rejects_bad_params(esrgan, bad):
    model, session = esrgan
    with pytest.raises(ValueError):
        run_restorer(model, np.zeros((8, 8), np.float32), bad)
    assert session.feeds == []


def test_missing_onnx_explains_export(monkeypatch, tmp_path):
    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"tools/export_onnx/realesrgan\.py"):
        RealESRGAN().load(CPU)
    assert realesrgan.ONNX not in _onnx.ONNX_SHA256  # not hosted yet


class EchoSession:
    """Identity network that records input shapes (NAFNet keeps the size)."""

    def __init__(self):
        self.shapes = []

    def run(self, _, feeds):
        x = feeds["image"]
        self.shapes.append(x.shape)
        return [x * 0.5]


def test_nafnet_pads_to_multiple_of_16_and_tiles(monkeypatch, tmp_path):
    from colorizer.models import nafnet

    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    for v in nafnet.VARIANTS.values():
        (tmp_path / v.onnx).touch()
    sessions = {}

    def fake_create(path, device):
        sessions[path.name] = EchoSession()
        return sessions[path.name], device

    monkeypatch.setattr(nafnet, "create_session", fake_create)
    model = nafnet.NAFNet()
    model.load(CPU)
    gray = np.random.default_rng(2).uniform(0, 1, (500, 403)).astype(np.float32)
    out = run_restorer(model, gray, {"variant": "deblur"})
    assert out.shape == gray.shape and np.allclose(out, gray * 0.5)
    shapes = sessions["nafnet_deblur.onnx"].shapes
    assert len(shapes) > 1  # tiled
    assert all(s[2] % 16 == 0 and s[3] % 16 == 0 and s[1] == 3 for s in shapes)
    run_restorer(model, gray, {"variant": "denoise"})
    assert set(sessions) == {"nafnet_deblur.onnx", "nafnet_denoise.onnx"}
    with pytest.raises(ValueError):
        run_restorer(model, gray, {"variant": "sharpen"})
