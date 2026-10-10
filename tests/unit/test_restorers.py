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


def test_missing_onnx_downloads_from_models_v2(monkeypatch, tmp_path):
    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    fetched = []

    def fake_ensure(w):
        fetched.append(w)
        raise OSError("offline")

    monkeypatch.setattr(_onnx, "ensure", fake_ensure)
    with pytest.raises(OSError):
        RealESRGAN().load(CPU)
    assert fetched[0].url.endswith("/models-v2/realesrgan_general_x4v3.onnx")


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


def test_paste_face_blends_only_inside_the_face():
    import cv2

    from colorizer.models.codeformer import FACE_SIZE, paste_face

    canvas = np.zeros((300, 400))
    face = np.ones((FACE_SIZE, FACE_SIZE), np.float32)
    # Face aligned from a 128 px square at (100, 50): scale 4 into the 512 template.
    affine = np.array([[4.0, 0, -400], [0, 4.0, -200]], np.float32)
    out = paste_face(canvas, face, cv2.invertAffineTransform(affine), 1)
    assert out[50 + 64, 100 + 64] == pytest.approx(1.0)  # centre: restored face
    assert out[10, 10] == 0.0 and out[250, 350] == 0.0  # outside: untouched
    edge = out[50 + 64, 100 : 100 + 64]  # from the face's left border to its centre
    assert edge[0] == 0.0 and edge[-1] == pytest.approx(1.0)
    assert ((edge > 0.05) & (edge < 0.95)).sum() >= 3  # feathered, not a hard step
    up = paste_face(np.zeros((600, 800)), face, cv2.invertAffineTransform(affine), 2)
    assert up[2 * 114, 2 * 164] == pytest.approx(1.0)


@pytest.mark.parametrize("stub_id", ["swinir", "seedvr2"])
def test_stubs_registered_but_disabled(stub_id):
    from colorizer.core.registry import Registry
    from colorizer.core.restore import validate_restore_params

    cls = Registry().restorers.get_class(stub_id)
    assert cls.enabled is False and cls.license.startswith("Apache-2.0")
    with pytest.raises(ValueError, match="not implemented"):
        validate_restore_params(cls, {})


def test_codeformer_declares_noncommercial_warning():
    from colorizer.models.codeformer import CodeFormer

    assert "non-commercial" in CodeFormer.warning
    assert {p.name: p.default for p in CodeFormer.params} == {"fidelity": 0.7, "upscale_bg": False}


def test_nafnet_divergence_returns_input_with_warning(monkeypatch, tmp_path, caplog):
    from colorizer.models import nafnet

    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    (tmp_path / "nafnet_denoise.onnx").touch()

    class Exploding(EchoSession):
        def run(self, _, feeds):
            x = feeds["image"]
            self.shapes.append(x.shape)
            # The second tile blows up, like the real networks on out-of-domain input.
            return [x * (200.0 if len(self.shapes) == 2 else 0.5)]

    session = Exploding()
    monkeypatch.setattr(nafnet, "create_session", lambda path, device: (session, device))
    model = nafnet.NAFNet()
    model.load(CPU)
    gray = np.random.default_rng(3).uniform(0.2, 0.8, (500, 500)).astype(np.float32)
    with caplog.at_level("WARNING"):
        out = run_restorer(model, gray, {"variant": "denoise"})
    assert np.array_equal(out, gray)  # unchanged, not a patchwork or garbage
    assert len(session.shapes) == 2  # stopped at the first diverged tile
    assert "left unchanged" in caplog.text
