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
    """Same-size network (x 0.5) that records input shapes."""

    def __init__(self):
        self.shapes = []

    def run(self, _, feeds):
        x = feeds["image"]
        self.shapes.append(x.shape)
        return [x * 0.5]


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


def test_scunet_runs_every_tile_at_one_shape(monkeypatch, tmp_path):
    from colorizer.models import scunet

    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    for name in scunet.ONNX.values():
        (tmp_path / name).touch()
    sessions = {}

    def fake_create(path, device):
        sessions[path.name] = EchoSession()
        return sessions[path.name], device

    monkeypatch.setattr(scunet, "create_session", fake_create)
    model = scunet.SCUNet()
    model.load(CPU)
    gray = np.random.default_rng(2).uniform(0, 1, (700, 333)).astype(np.float32)
    out = run_restorer(model, gray, {"variant": "psnr"})
    assert out.shape == gray.shape and np.allclose(out, gray * 0.5)
    shapes = set(sessions["scunet_real_psnr.onnx"].shapes)
    assert shapes == {(1, 3, scunet.RUN_SIZE, scunet.RUN_SIZE)}  # constant memory
    run_restorer(model, np.zeros((40, 50), np.float32), {"variant": "gan"})
    assert sessions["scunet_real_gan.onnx"].shapes == [(1, 3, 320, 320)]
    with pytest.raises(ValueError):
        run_restorer(model, gray, {"variant": "sharp"})


class FakeDetector:
    """Detection 'network': marks very dark pixels (below -0.8 in [-1, 1] input) as damage."""

    def __init__(self):
        self.shapes = []

    def run(self, _, feeds):
        x = feeds["gray"]
        self.shapes.append(x.shape)
        return [np.where(x < -0.8, 10.0, -10.0).astype(np.float32)]


@pytest.fixture
def descratch(monkeypatch, tmp_path):
    from colorizer.models import descratch

    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    (tmp_path / descratch.ONNX).touch()
    detector = FakeDetector()
    monkeypatch.setattr(descratch, "create_session", lambda path, device: (detector, device))
    model = descratch.Descratch()
    model.load(CPU)
    return model, detector


def test_descratch_fills_detected_damage(descratch):
    model, detector = descratch
    gray = np.full((300, 520), 0.6, np.float32)
    gray[150:153, 40:480] = 0.0  # a dark scratch
    out = run_restorer(model, gray, {"grow": 1})
    assert out.shape == gray.shape and out.dtype == np.float32
    assert abs(out[151, 260] - 0.6) < 0.02  # filled from the surroundings
    assert np.allclose(out[:100], 0.6, atol=1e-4)  # undamaged area untouched
    (shape,) = detector.shapes
    assert min(shape[2:]) == 256 and shape[2] % 16 == 0 and shape[3] % 16 == 0
    run_restorer(model, gray, {"detail": "fine"})
    assert min(detector.shapes[-1][2:]) == 512


def test_descratch_without_damage_is_identity(descratch):
    model, _ = descratch
    gray = np.random.default_rng(4).uniform(0.3, 0.9, (64, 80)).astype(np.float32)
    assert np.array_equal(run_restorer(model, gray, {}), gray)
    assert np.array_equal(run_restorer(model, gray * 0 + 0.0, {"sensitivity": 0.0}), gray * 0)


@pytest.mark.parametrize("bad", [{"sensitivity": 1.5}, {"grow": 11}, {"detail": "max"}])
def test_descratch_rejects_bad_params(descratch, bad):
    model, detector = descratch
    with pytest.raises(ValueError):
        run_restorer(model, np.zeros((32, 32), np.float32), bad)
    assert detector.shapes == []
