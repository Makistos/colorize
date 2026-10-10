import numpy as np
import pytest

from colorizer.core import weights
from colorizer.core.runtime import CPU
from colorizer.models import _onnx
from colorizer.models.ddcolor import DDColor
from colorizer.models.deoldify import DeOldify


class FakeSession:
    """Records inputs; returns ab (2 channels) or RGB (3 channels) of the input's size."""

    def __init__(self, out_channels, value):
        self.out_channels, self.value, self.inputs = out_channels, value, []

    def run(self, _, feeds):
        (x,) = feeds.values()
        self.inputs.append(x)
        n, _, h, w = x.shape
        return [np.full((n, self.out_channels, h, w), self.value, np.float32)]


@pytest.fixture
def L():
    return np.linspace(0, 100, 300 * 200, dtype=np.float32).reshape(300, 200)


def loaded(model, session):
    model.load(CPU)
    model._sessions = {v: session for v in model.variants}
    return model


def test_ddcolor_feeds_gray_rgb_at_working_size(L):
    sess = FakeSession(2, 12.0)
    model = loaded(DDColor(), sess)
    ab = model.predict_ab(L, **model.validate_params({"size": 320}))
    (x,) = sess.inputs
    assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0
    np.testing.assert_allclose(x[0, 0], x[0, 1], atol=1e-4)  # neutral gray
    assert ab.shape == (320, 320, 2) and np.all(ab == 12.0)


def test_deoldify_converts_rgb_output_to_ab(L):
    sess = FakeSession(3, 0.0)
    sess.run = lambda _, feeds: [
        np.stack([np.full((112, 112), v, np.float32) for v in (0.8, 0.5, 0.2)])[None]
    ]
    model = loaded(DeOldify(), sess)
    ab = model.predict_ab(L, **model.validate_params({"render_factor": 7}))
    assert ab.shape == (112, 112, 2)
    assert ab[..., 0].mean() > 5 and ab[..., 1].mean() > 20  # orange: +a, +b


def test_deoldify_working_size_from_render_factor(L):
    sess = FakeSession(3, 0.5)
    model = loaded(DeOldify(), sess)
    model.predict_ab(L, **model.validate_params({}))
    assert sess.inputs[0].shape == (1, 3, 35 * 16, 35 * 16)


@pytest.mark.parametrize(
    ("cls", "bad"),
    [
        (DDColor, {"size": 300}),
        (DDColor, {"variant": "huge"}),
        (DeOldify, {"render_factor": 50}),
        (DeOldify, {"variant": "video"}),
    ],
)
def test_param_validation(cls, bad):
    with pytest.raises(ValueError):
        cls.validate_params(bad)


@pytest.mark.parametrize("cls", [DDColor, DeOldify])
def test_missing_onnx_explains_export(cls, tmp_path, monkeypatch, L):
    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    monkeypatch.setattr(_onnx, "ONNX_SHA256", {})  # no hosted build
    model = cls()
    model.load(CPU)
    with pytest.raises(FileNotFoundError, match="tools/export_onnx"):
        model.predict_ab(L, **model.validate_params({}))


def test_missing_onnx_downloads_hosted_build(tmp_path, monkeypatch):
    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    fetched = []
    monkeypatch.setattr(_onnx, "ensure", lambda w: fetched.append(w) or tmp_path / w.filename)
    assert _onnx.ensure_onnx("ddcolor_tiny.onnx", "hint") == tmp_path / "ddcolor_tiny.onnx"
    (w,) = fetched
    assert w.url == _onnx.RELEASES_URL + "models-v1/ddcolor_tiny.onnx"
    assert _onnx.onnx_url("codeformer.onnx").endswith("/models-v2/codeformer.onnx")
    assert w.sha256 == _onnx.ONNX_SHA256["ddcolor_tiny.onnx"]


def test_local_export_skips_download(tmp_path, monkeypatch):
    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    monkeypatch.setattr(_onnx, "ensure", lambda w: pytest.fail("downloaded"))
    (tmp_path / "ddcolor_tiny.onnx").touch()
    assert _onnx.ensure_onnx("ddcolor_tiny.onnx", "hint") == tmp_path / "ddcolor_tiny.onnx"


def test_every_onnx_variant_is_hosted():
    from colorizer.models import codeformer, descratch, realesrgan, scunet, zhang

    names = {v.onnx for cls in (DDColor, DeOldify) for v in cls.variants.values()}
    names |= {zhang.ECCV16_ONNX, zhang.SIGGRAPH17_ONNX, realesrgan.ONNX, codeformer.ONNX}
    names |= {descratch.ONNX, *scunet.ONNX.values()}
    assert names == set(_onnx.ONNX_SHA256)
    assert set(_onnx.ONNX_RELEASE) <= names


def test_sessions_created_per_variant_on_demand(tmp_path, monkeypatch, L):
    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    created = []

    def fake_create(path, device):
        created.append(path.name)
        return FakeSession(3, 0.5), device

    monkeypatch.setattr(_onnx, "create_session", fake_create)
    for v in DeOldify.variants.values():
        (tmp_path / v.onnx).touch()
    model = DeOldify()
    model.load(CPU)
    for variant in ("stable", "stable", "artistic"):
        model.predict_ab(L, **model.validate_params({"variant": variant, "render_factor": 7}))
    assert created == ["deoldify_stable.onnx", "deoldify_artistic.onnx"]
    model.unload()
    assert model._sessions == {}


def test_siggraph_hint_encoding():
    from colorizer.models.zhang import encode_hints

    ab, mask = encode_hints([(50, 20, (255, 0, 0)), (999, 5, (0, 0, 255))], (40, 100), 256)
    assert ab.shape == (2, 256, 256) and mask.shape == (1, 256, 256)
    cy, cx = int(20 * 256 / 40), int(50 * 256 / 100)
    assert mask[0, cy, cx] == 1 and mask.sum() == 9  # one 3x3 patch; off-image point ignored
    assert ab[0, cy, cx] > 50 and ab[1, cy, cx] > 40  # red: +a, +b
    assert np.all(ab[:, mask[0] == 0] == 0)
