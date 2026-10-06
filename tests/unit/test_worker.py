import logging
import threading

import pytest

from colorizer.core.worker import Cancelled, Worker


@pytest.fixture
def worker():
    w = Worker()
    yield w
    w.shutdown()


def test_result_and_progress(worker):
    def fn(ctx):
        ctx.progress(0.5, "half")
        return 42

    job = worker.submit(fn)
    assert job.result(timeout=5) == 42
    assert job.progress == (0.5, "half")


def test_exception_propagates(worker):
    def fn(ctx):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        worker.submit(fn).result(timeout=5)


def test_cancel_running_job_stops_at_next_step(worker):
    started, release = threading.Event(), threading.Event()
    steps = []

    def fn(ctx):
        started.set()
        release.wait(5)
        steps.append(1)
        ctx.progress(0.5, "step 2")  # raises Cancelled
        steps.append(2)

    job = worker.submit(fn)
    started.wait(5)
    job.cancel()
    release.set()
    with pytest.raises(Cancelled):
        job.result(timeout=5)
    worker.submit(lambda ctx: None).result(timeout=5)  # worker drained
    assert steps == [1]


def test_cancel_queued_job_never_runs(worker):
    gate = threading.Event()
    ran = []
    first = worker.submit(lambda ctx: gate.wait(5))
    second = worker.submit(lambda ctx: ran.append(1))
    second.cancel()
    gate.set()
    first.result(timeout=5)
    with pytest.raises(Cancelled):
        second.result(timeout=5)
    worker.submit(lambda ctx: None).result(timeout=5)
    assert ran == []


def test_warnings_collected_only_from_job(worker):
    log = logging.getLogger("colorizer.test")

    def fn(ctx):
        log.warning("GPU failed, using CPU")
        log.info("not a warning")

    job = worker.submit(fn)
    log.warning("from another thread")
    job.result(timeout=5)
    assert job.warnings == ["GPU failed, using CPU"]


def test_wait_timeout(worker):
    gate = threading.Event()
    job = worker.submit(lambda ctx: gate.wait(5))
    assert job.wait(0.01) is False
    gate.set()
    assert job.wait(5) is True
