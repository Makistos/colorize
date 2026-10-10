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


RESTORERS = {
    "realesrgan": ({"scale": 2, "tile": 32}, 2),
    "descratch": ({}, 1),
    "descratch-fine": ({"detail": "fine", "grow": 0}, 1),
    "scunet": ({"variant": "psnr"}, 1),
    "scunet-gan": ({"variant": "gan"}, 1),
    "codeformer": ({"fidelity": 0.7}, 1),
    "codeformer-bg": ({"upscale_bg": True}, 2),
}


@pytest.mark.parametrize("restorer_id", list(RESTORERS))
def test_golden_restorer(registry, restorer_id):
    import cv2

    from colorizer.core.pipeline import L_to_gray
    from colorizer.core.restore import run_restorer

    params, scale = RESTORERS[restorer_id]
    restorer = registry.restorers.get(restorer_id.split("-")[0], CPU)
    gray = L_to_gray(load_image(FIXTURE).L)
    out = run_restorer(restorer, gray, params)
    assert out.dtype == np.float32 and out.shape == (64 * scale, 96 * scale)
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
    # Restored, not replaced: still closely resembles the input.
    up = cv2.resize(gray, out.shape[::-1], interpolation=cv2.INTER_CUBIC)
    assert np.corrcoef(up.ravel(), out.ravel())[0, 1] > 0.9


def test_codeformer_restores_a_face(registry):
    """Detects the face in a real portrait, sharpens it, and leaves the rest alone."""
    import cv2
    from skimage import data

    restorer = registry.restorers.get("codeformer", CPU)
    gray = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    blurred = np.clip(cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC), 0, 1)
    out = restorer.restore(blurred, fidelity=0.7, upscale_bg=False)

    def sharpness(img):
        return cv2.Laplacian(img[30:230, 150:330].astype(np.float64), cv2.CV_64F).var()

    assert sharpness(out) > 1.5 * sharpness(blurred)
    assert np.allclose(out[400:], blurred[400:], atol=1e-6)


@pytest.mark.parametrize("variant", ["psnr", "gan"])
def test_scunet_removes_grain(registry, variant):
    import cv2
    from skimage import data

    from colorizer.core.restore import run_restorer

    restorer = registry.restorers.get("scunet", CPU)
    clean = data.camera()[:384, :448].astype(np.float32) / 255
    grain = np.random.default_rng(0).normal(0, 0.06, clean.shape)
    grainy = np.clip(clean + grain, 0, 1).astype(np.float32)
    _, enc = cv2.imencode(
        ".jpg", np.round(grainy * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 70]
    )
    grainy = cv2.imdecode(enc, 0).astype(np.float32) / 255
    out = run_restorer(restorer, grainy, {"variant": variant})
    assert np.mean((out - clean) ** 2) < 0.5 * np.mean((grainy - clean) ** 2)


def test_descratch_removes_scratches(registry):
    """Thin light scratches drawn on a real photo are found and filled in."""
    import cv2
    from skimage import data

    from colorizer.core.restore import run_restorer

    restorer = registry.restorers.get("descratch", CPU)
    clean = data.camera()[:384, :448].copy()
    damaged = clean.copy()
    # Light damage (the detector's domain; it does not find dark scratches).
    for (x0, y0, x1, y1), value in (((10, 20, 430, 300), 245), ((300, 10, 80, 370), 225)):
        cv2.line(damaged, (x0, y0), (x1, y1), value, 2)
    clean_f, damaged_f = clean.astype(np.float32) / 255, damaged.astype(np.float32) / 255
    out = run_restorer(restorer, damaged_f, {})
    before, after = np.mean((damaged_f - clean_f) ** 2), np.mean((out - clean_f) ** 2)
    assert after < 0.3 * before, (before, after)
