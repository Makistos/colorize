import numpy as np
import pytest
from PIL import Image

from colorizer.core.batch import BatchItem, run_batch
from colorizer.core.postprocess import Postprocess
from colorizer.core.runtime import CPU
from colorizer.core.worker import Cancelled, Worker
from tests.conftest import DummyModel


def make_items(tmp_path, n):
    items = []
    for i in range(n):
        src = tmp_path / f"in{i}.png"
        Image.fromarray(np.full((10, 10), 100, np.uint8)).save(src)
        items.append(BatchItem(src, tmp_path / "out" / f"o{i}.png", "png"))
    return items


def model():
    m = DummyModel()
    m.load(CPU)
    return m


def test_model_not_loaded_when_everything_exists(tmp_path):
    items = make_items(tmp_path, 1)
    items[0].dst.parent.mkdir()
    items[0].dst.write_bytes(b"x")
    result = run_batch(items, lambda: pytest.fail("loaded"), {}, Postprocess())
    assert result.summary() == "0 colorized, 1 skipped, 0 failed"


def test_cancel_between_files(tmp_path):
    items = make_items(tmp_path, 5)
    worker = Worker()
    seen = []

    def fn(ctx):
        def get():
            return model()

        original = ctx.progress

        def progress(fraction, message=""):
            seen.append(message)
            if len(seen) == 3:  # loading, file 1, file 2 -> cancel before file 2 runs
                job.cancel()
            original(fraction, message)

        ctx.progress = progress
        return run_batch(items, get, {}, Postprocess(), ctx=ctx)

    job = worker.submit(fn)
    with pytest.raises(Cancelled):
        job.result(timeout=10)
    worker.shutdown()
    assert [p.exists() for p in (i.dst for i in items)] == [True, False, False, False, False]


def test_batch_runs_restore_chain(tmp_path):
    from colorizer.core.restore import RestoreStep
    from tests.conftest import DummyRestorer

    items = make_items(tmp_path, 2)
    loaded = []

    def get_restorer(rid):
        loaded.append(rid)
        return DummyRestorer()

    result = run_batch(
        items,
        model,
        {},
        Postprocess(),
        restore=[RestoreStep("dummy_restore", {"scale": 2})],
        get_restorer=get_restorer,
    )
    assert len(result.done) == 2 and loaded == ["dummy_restore"]
    assert Image.open(items[0].dst).size == (20, 20)


def test_batch_restore_needs_a_loader(tmp_path):
    from colorizer.core.restore import RestoreStep

    with pytest.raises(ValueError, match="get_restorer"):
        run_batch(make_items(tmp_path, 1), model, {}, Postprocess(), restore=[RestoreStep("x")])
