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


@pytest.mark.parametrize("model_id", list(BUILTIN_MODELS))
def test_golden(registry, model_id):
    model = registry.get(model_id, CPU)
    src = load_image(FIXTURE)
    ab = infer_ab(src, model, {})
    assert ab.dtype == np.float32 and ab.ndim == 3 and ab.shape[2] == 2
    assert np.isfinite(ab).all()
    rgb = render(src, ab)
    assert rgb.shape == (64, 96, 3)
    chroma = np.hypot(ab[..., 0], ab[..., 1])
    assert chroma.mean() > 2.0, "model produced (near) grayscale output"
