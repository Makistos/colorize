"""Real-model golden tests. Run with COLORIZER_TEST_WEIGHTS=1."""

from pathlib import Path

import numpy as np
import pytest

from colorizer.core.pipeline import infer_ab, load_image, render
from colorizer.core.registry import BUILTIN_MODELS, Registry
from colorizer.core.runtime import CPU

FIXTURE = Path(__file__).parent.parent / "fixtures" / "bw_96x64.png"

pytestmark = pytest.mark.weights


@pytest.fixture(scope="module")
def registry():
    reg = Registry(max_loaded=1)
    yield reg
    reg.unload_all()


FAST_MODELS = [m for m in BUILTIN_MODELS if m != "sd_controlnet"]


@pytest.mark.parametrize("model_id", FAST_MODELS)
def test_golden(registry, model_id):
    check_golden(registry, model_id, {})


@pytest.mark.diffusion
def test_golden_sd_controlnet(registry):
    pytest.importorskip("diffusers")
    check_golden(registry, "sd_controlnet", {"steps": 10, "seed": 1})


def check_golden(registry, model_id, params):
    model = registry.get(model_id, CPU)
    src = load_image(FIXTURE)
    ab = infer_ab(src, model, params)
    assert ab.dtype == np.float32 and ab.ndim == 3 and ab.shape[2] == 2
    assert np.isfinite(ab).all()
    rgb = render(src, ab)
    assert rgb.shape == (64, 96, 3)
    chroma = np.hypot(ab[..., 0], ab[..., 1])
    # A broken model gives ~0 everywhere; real ones colour at least some regions clearly.
    assert np.percentile(chroma, 90) > 3.0, "model produced (near) grayscale output"
