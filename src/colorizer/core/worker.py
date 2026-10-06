"""Background job execution with progress reporting and cooperative cancellation.

Jobs run one at a time on a single worker thread: models are not thread-safe and usually
share one accelerator. Cancellation is checked between steps; a running model inference
call is not interrupted.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import CancelledError as FutureCancelled
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Generic, TypeVar

T = TypeVar("T")

log = logging.getLogger(__name__)

# Warnings logged under this logger while a job runs are collected on the job.
CAPTURE_LOGGER = "colorizer"


class Cancelled(Exception):
    """Raised inside a job when it has been cancelled, and by ``Job.result()``."""


class JobContext:
    """Passed to the job function to report progress and check for cancellation."""

    def __init__(self, job: Job[object]) -> None:
        self._job = job

    def progress(self, fraction: float, message: str = "") -> None:
        """Record progress (0..1) and raise ``Cancelled`` if the job was cancelled."""
        self._job._set_progress(min(max(fraction, 0.0), 1.0), message)
        self.check_cancelled()

    def check_cancelled(self) -> None:
        if self._job.cancel_requested:
            raise Cancelled()


class Job(Generic[T]):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._fraction = 0.0
        self._message = "queued"
        self.warnings: list[str] = []
        self.future: Future[T] = Future()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def progress(self) -> tuple[float, str]:
        with self._lock:
            return self._fraction, self._message

    def _set_progress(self, fraction: float, message: str) -> None:
        with self._lock:
            self._fraction = fraction
            if message:
                self._message = message

    def cancel(self) -> None:
        """Request cancellation. A queued job never starts; a running one stops at its next
        progress/check call."""
        self._cancel.set()
        self.future.cancel()

    def done(self) -> bool:
        return self.future.done()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait up to ``timeout`` seconds; return whether the job finished."""
        try:
            self.future.exception(timeout=timeout)
        except TimeoutError:
            return False
        except FutureCancelled:
            pass
        return True

    def result(self, timeout: float | None = None) -> T:
        """Return the result, re-raising the job's exception, or ``Cancelled``."""
        try:
            return self.future.result(timeout=timeout)
        except FutureCancelled:
            raise Cancelled() from None


class _WarningCollector(logging.Handler):
    def __init__(self, sink: list[str], thread_id: int) -> None:
        super().__init__(level=logging.WARNING)
        self._sink = sink
        self._thread_id = thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self._thread_id:
            self._sink.append(record.getMessage())


class Worker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="colorizer-job")

    def submit(self, fn: Callable[[JobContext], T]) -> Job[T]:
        job: Job[T] = Job()
        inner = self._executor.submit(self._run, job, fn)
        # Mirror the executor future onto the job's own future so Job.cancel() can mark a
        # queued job cancelled immediately.
        inner.add_done_callback(lambda f: _copy_result(f, job))
        return job

    @staticmethod
    def _run(job: Job[T], fn: Callable[[JobContext], T]) -> T:
        if job.cancel_requested:
            raise Cancelled()
        logger = logging.getLogger(CAPTURE_LOGGER)
        collector = _WarningCollector(job.warnings, threading.get_ident())
        logger.addHandler(collector)
        try:
            job._set_progress(0.0, "running")
            return fn(JobContext(job))  # type: ignore[arg-type]
        finally:
            logger.removeHandler(collector)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


def _copy_result(src: Future[T], job: Job[T]) -> None:
    dst = job.future
    if dst.done():  # cancelled by the user while queued or running
        return
    if src.cancelled():
        dst.cancel()
        return
    exc = src.exception()
    if isinstance(exc, Cancelled):
        dst.cancel()
    elif exc is not None:
        dst.set_exception(exc)
    else:
        dst.set_result(src.result())
