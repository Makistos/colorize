import numpy as np
import pytest

from colorizer.core import postprocess
from colorizer.core.pipeline import SourceImage, render
from colorizer.core.postprocess import Postprocess

rng = np.random.default_rng(0)
AB = rng.uniform(-60, 60, (24, 32, 2)).astype(np.float32)


def test_identity():
    np.testing.assert_allclose(postprocess.apply(AB, Postprocess()), AB, atol=1e-6)


@pytest.mark.parametrize("post", [Postprocess(saturation=0.0), Postprocess(chroma_blend=0.0)])
def test_zero_gives_gray(post):
    assert np.all(postprocess.apply(AB, post) == 0)


def test_saturation_and_blend_scale():
    np.testing.assert_allclose(
        postprocess.apply(AB, Postprocess(saturation=2.0)), AB * 2, rtol=1e-6
    )
    np.testing.assert_allclose(
        postprocess.apply(AB, Postprocess(chroma_blend=0.25)), AB / 4, rtol=1e-6
    )


def test_temperature_and_tint_signs():
    zero = np.zeros_like(AB)
    warm = postprocess.apply(zero, Postprocess(temperature=1.0))
    assert np.all(warm[..., 1] > 0) and np.all(warm[..., 0] == 0)  # +b = yellow
    magenta = postprocess.apply(zero, Postprocess(tint=1.0))
    assert np.all(magenta[..., 0] > 0) and np.all(magenta[..., 1] == 0)  # +a = magenta
    cool = postprocess.apply(zero, Postprocess(temperature=-1.0))
    assert np.all(cool[..., 1] < 0)


def test_smooth_reduces_variance_and_keeps_mean():
    out = postprocess.apply(AB, Postprocess(ab_smooth=2.0))
    assert out.std() < AB.std() / 2
    assert abs(out.mean() - AB.mean()) < 1.0


def test_smooth_never_touches_L():
    src = SourceImage.from_rgb(rng.uniform(0, 1, (24, 32)))
    rgb = render(src, AB, Postprocess(ab_smooth=3.0))
    L_back = SourceImage.from_rgb(rgb).L
    # Only gamut clipping may move L; most pixels must be unchanged.
    assert np.median(np.abs(L_back - src.L)) < 0.5


@pytest.mark.parametrize("values", [{"saturation": 3}, {"tint": -2}, {"bogus": 1}])
def test_from_mapping_validates(values):
    with pytest.raises(ValueError):
        Postprocess.from_mapping(values)
