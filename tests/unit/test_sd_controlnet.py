import numpy as np
import pytest

from colorizer.core.worker import Cancelled
from colorizer.models.sd_controlnet import SDControlNet, working_size

pytest.importorskip("torch")


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, **kw):
        self.calls.append((prompt, kw))
        for step in range(int(kw["num_inference_steps"] * kw["strength"])):
            kw["callback_on_step_end"](self, step, None, {})
        h, w = kw["image"].size[1], kw["image"].size[0]
        out = np.zeros((h, w, 3), np.float32)
        out[...] = (0.2, 0.4, 0.8)  # blue
        return type("R", (), {"images": [out]})()


def model_with(pipe):
    m = SDControlNet()
    m._pipe = pipe
    return m


def test_working_size_keeps_aspect_in_multiples_of_64():
    assert working_size(1004, 935) == (512, 448)
    assert working_size(100, 2000) == (64, 512)


def test_predict_ab_runs_pipe_and_reports_steps():
    pipe = FakePipe()
    m = model_with(pipe)
    seen = []
    m.step_callback = seen.append
    L = np.full((300, 200), 50.0, np.float32)
    ab = m.predict_ab(L, **SDControlNet.validate_params({"steps": 10, "seed": 3}))
    prompt, kw = pipe.calls[0]
    assert "photograph" in prompt and kw["image"].size == (320, 512)  # PIL size is (w, h)
    assert kw["image"] is kw["control_image"] and kw["guidance_scale"] == 7.0
    assert ab.shape == (512, 320, 2) and ab[..., 1].mean() < -20  # blue: -b
    assert seen[-1] == 1.0 and len(seen) == 10


def test_cancel_propagates_mid_generation():
    m = model_with(FakePipe())

    def cancel_at_3(fraction):
        if fraction >= 0.3:
            raise Cancelled()

    m.step_callback = cancel_at_3
    with pytest.raises(Cancelled):
        m.predict_ab(np.zeros((64, 64), np.float32), **SDControlNet.validate_params({}))


def test_zero_strength_stays_gray():
    pipe = FakePipe()
    ab = model_with(pipe).predict_ab(
        np.zeros((64, 64), np.float32), **SDControlNet.validate_params({"strength": 0.0})
    )
    assert not pipe.calls and np.all(ab == 0)
