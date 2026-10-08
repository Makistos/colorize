import numpy as np
import pytest
from PIL import Image

from colorizer.core import pipeline
from colorizer.core.pipeline import SourceImage, infer_ab, load_image, render, save_image
from colorizer.core.runtime import CPU
from tests.conftest import DummyModel


@pytest.fixture
def model():
    m = DummyModel()
    m.load(CPU)
    return m


def gradient(w=60, h=40):
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))


def test_exif_orientation_applied(tmp_path):
    im = Image.fromarray(gradient())  # 60 wide, 40 high
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° CW on display
    exif[0x010F] = "TestCam"
    im.save(tmp_path / "rot.jpg", exif=exif.tobytes())
    src = load_image(tmp_path / "rot.jpg")
    assert src.size == (60, 40)
    saved = Image.Exif()
    saved.load(src.exif)
    assert saved.get(0x0112, 1) == 1 and saved[0x010F] == "TestCam"


def test_end_to_end_png_keeps_size_and_L(tmp_path, model):
    Image.fromarray(gradient()).save(tmp_path / "in.png")
    pipeline.colorize_file(tmp_path / "in.png", tmp_path / "out.png", model, {"size": 16})
    out = np.asarray(Image.open(tmp_path / "out.png"))
    assert out.shape == (40, 60, 3) and out.dtype == np.uint8
    assert model.calls[0] == {"size": 16, "a": 10.0, "b": -20.0}
    # Coloured, but lightness preserved.
    assert np.ptp(out.astype(int), axis=2).max() > 10
    L_in = SourceImage.from_rgb(gradient()).L
    L_out = SourceImage.from_rgb(out).L
    assert np.median(np.abs(L_out - L_in)) < 1.0


def test_rgb_input_treated_as_gray(tmp_path, model):
    rgb = np.dstack([gradient(), gradient() // 2, np.zeros_like(gradient())])
    Image.fromarray(rgb).save(tmp_path / "c.png")
    src = load_image(tmp_path / "c.png")
    assert src.L.ndim == 2 and src.bit_depth == 8


def test_invalid_params_raise_before_inference(model):
    src = SourceImage.from_rgb(gradient())
    with pytest.raises(ValueError):
        infer_ab(src, model, {"size": 17})
    with pytest.raises(ValueError):
        infer_ab(src, model, {"unknown": 1})
    assert model.calls == []


def test_bad_model_output_shape(model, monkeypatch):
    monkeypatch.setattr(model, "predict_ab", lambda L, **p: np.zeros((4, 4), np.float32))
    with pytest.raises(ValueError, match="shape"):
        infer_ab(SourceImage.from_rgb(gradient()), model, {})


def test_jpeg_quality_and_exif(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 1, (64, 64, 3))
    exif = Image.Exif()
    exif[0x010F] = "TestCam"
    save_image(rgb, tmp_path / "hi.jpg", "jpg", quality=95, exif=exif.tobytes())
    save_image(rgb, tmp_path / "lo.jpg", "jpg", quality=20)
    assert (tmp_path / "hi.jpg").stat().st_size > (tmp_path / "lo.jpg").stat().st_size
    assert Image.open(tmp_path / "hi.jpg").getexif()[0x010F] == "TestCam"


@pytest.mark.parametrize("fmt", ["png", "tiff"])
def test_exif_preserved_8bit(tmp_path, fmt):
    exif = Image.Exif()
    exif[0x010F] = "TestCam"
    path = tmp_path / f"o.{fmt}"
    save_image(np.full((8, 8, 3), 0.5), path, fmt, exif=exif.tobytes())
    assert Image.open(path).getexif()[0x010F] == "TestCam"


def test_16bit_tiff_roundtrip(tmp_path, model):
    gray16 = np.tile(np.linspace(0, 65535, 50).astype(np.uint16), (30, 1))
    Image.fromarray(gray16).save(tmp_path / "in.tif")
    src = load_image(tmp_path / "in.tif")
    assert src.bit_depth == 16
    pipeline.colorize_file(tmp_path / "in.tif", tmp_path / "out.tiff", model, {}, fmt="tiff")
    import cv2

    out = cv2.imread(str(tmp_path / "out.tiff"), cv2.IMREAD_UNCHANGED)
    assert out.dtype == np.uint16 and out.shape == (30, 50, 3)


def test_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        save_image(np.zeros((2, 2, 3)), tmp_path / "x.gif", "gif")  # type: ignore[arg-type]


def test_render_postprocess_does_not_need_model():
    src = SourceImage.from_rgb(gradient())
    ab = np.full((8, 8, 2), 30.0, np.float32)
    from colorizer.core.postprocess import Postprocess

    gray = render(src, ab, Postprocess(chroma_blend=0.0))
    assert np.allclose(gray[..., 0], gray[..., 2], atol=1e-4)


@pytest.mark.parametrize("ab_value", [(0.0, 0.0), (10.0, -20.0), (40.0, 60.0), (-128.0, 127.0)])
def test_output_L_is_the_full_resolution_input_L(ab_value):
    """The colorized image keeps the source's L, also where colours are out of gamut."""
    from skimage.color import rgb2lab

    rng = np.random.default_rng(0)
    src = SourceImage.from_rgb(rng.uniform(0, 1, (203, 301)))  # odd, non-square
    ab = np.empty((16, 24, 2), np.float32)  # model working size, much smaller
    ab[...] = ab_value
    out = render(src, ab)
    assert out.shape == (203, 301, 3)
    assert np.abs(rgb2lab(out)[..., 0] - src.L).max() < 1e-2


def test_lab_to_rgb_matches_skimage_in_gamut():
    from skimage.color import rgb2lab

    rgb = np.random.default_rng(1).uniform(0, 1, (40, 30, 3))
    lab = rgb2lab(rgb)
    assert np.abs(pipeline.lab_to_rgb(lab[..., 0], lab[..., 1:]) - rgb).max() < 1e-5


def test_out_of_gamut_keeps_hue():
    L = np.full((1, 1), 50.0)
    ab = np.array([[[90.0, 60.0]]])
    from skimage.color import rgb2lab

    lab = rgb2lab(pipeline.lab_to_rgb(L, ab))[0, 0]
    assert abs(lab[0] - 50.0) < 1e-2
    assert np.hypot(lab[1], lab[2]) < np.hypot(90, 60)  # chroma reduced...
    assert abs(np.arctan2(lab[2], lab[1]) - np.arctan2(60, 90)) < 1e-2  # ...at the same hue


def test_ab_is_upscaled_bicubically():
    import cv2

    src = SourceImage.from_rgb(np.full((30, 50), 0.5))
    ab = np.random.default_rng(2).normal(0, 10, (6, 10, 2)).astype(np.float32)
    expected = pipeline.lab_to_rgb(src.L, cv2.resize(ab, (50, 30), interpolation=cv2.INTER_CUBIC))
    assert np.allclose(render(src, ab), expected)


def test_ab_smooth_defaults_to_off():
    from colorizer.core.postprocess import POSTPROCESS_PARAMS, Postprocess

    assert Postprocess().ab_smooth == 0.0
    assert {p.name: p.default for p in POSTPROCESS_PARAMS}["ab_smooth"] == 0.0


def test_restore_runs_before_colorizing_and_may_upscale(tmp_path, model):
    from tests.conftest import DummyRestorer

    Image.fromarray(gradient()).save(tmp_path / "in.png")
    restorer = DummyRestorer()
    pipeline.colorize_file(
        tmp_path / "in.png",
        tmp_path / "out.png",
        model,
        {},
        restore=[(restorer, {"scale": 2}), (restorer, {"offset": 0.1, "blend": 0.5})],
    )
    assert np.asarray(Image.open(tmp_path / "out.png")).shape == (80, 120, 3)
    assert restorer.calls == [{"scale": 2, "offset": 0.0}, {"scale": 1, "offset": 0.1}]


def test_restore_params_all_validated_before_any_step_runs():
    from tests.conftest import DummyRestorer

    restorer = DummyRestorer()
    src = SourceImage.from_rgb(gradient())
    with pytest.raises(ValueError):
        pipeline.restore_image(src, [(restorer, {}), (restorer, {"blend": 2.0})])
    assert restorer.calls == []
    assert pipeline.restore_image(src, []) is src


def test_restored_L_is_the_output_L():
    from tests.conftest import DummyRestorer

    src = SourceImage.from_rgb(gradient())
    restored = pipeline.restore_image(src, [(DummyRestorer(), {"offset": 0.2})])
    assert restored.size == src.size and restored.L.mean() > src.L.mean()
    from skimage.color import rgb2lab

    out = render(restored, np.full((4, 4, 2), 20.0, np.float32))
    assert np.abs(rgb2lab(out)[..., 0] - restored.L).max() < 1e-2


def test_gray_L_conversions_are_inverse():
    L = np.linspace(0, 100, 10001, dtype=np.float32)
    assert np.abs(pipeline.gray_to_L(pipeline.L_to_gray(L)) - L).max() < 1e-3
