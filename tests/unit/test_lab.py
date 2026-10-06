import numpy as np

from colorizer.core.pipeline import SourceImage, render


def test_gray_roundtrip_preserves_L_8bit():
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    src = SourceImage.from_rgb(gray)
    assert src.L.dtype == np.float32 and src.L.shape == (40, 50)
    assert src.L.min() >= 0 and src.L.max() <= 100
    rgb = render(src, np.zeros((8, 8, 2), np.float32))
    out = np.round(rgb * 255).astype(np.uint8)
    assert np.all(out[..., 0] == out[..., 1]) and np.all(out[..., 1] == out[..., 2])
    assert np.abs(out[..., 0].astype(int) - gray).max() <= 1


def test_gray_roundtrip_preserves_L_float():
    gray = np.linspace(0, 1, 64 * 64).reshape(64, 64)
    src = SourceImage.from_rgb(gray)
    rgb = render(src, np.zeros((64, 64, 2), np.float32))
    np.testing.assert_allclose(SourceImage.from_rgb(rgb).L, src.L, atol=1e-3)


def test_sepia_reduced_to_lightness():
    gray = np.full((4, 4), 0.5)
    sepia = np.dstack([gray * 1.0, gray * 0.85, gray * 0.65])
    L = SourceImage.from_rgb(sepia).L
    assert np.allclose(L, L[0, 0]) and 0 < L[0, 0] < 100


def test_render_upscales_ab_to_full_size():
    src = SourceImage.from_rgb(np.full((30, 70), 0.5))
    rgb = render(src, np.full((8, 8, 2), 20.0, np.float32))
    assert rgb.shape == (30, 70, 3)
    assert rgb[..., 0].mean() > rgb[..., 2].mean()  # +a,+b is warm
