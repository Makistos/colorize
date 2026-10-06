"""Colorize many files with one model and one set of settings (used by the CLI and the UI)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colorizer.core.base import ColorizerModel
from colorizer.core.pipeline import OutputFormat, colorize_file
from colorizer.core.postprocess import Postprocess
from colorizer.core.worker import JobContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchItem:
    src: Path
    dst: Path
    fmt: OutputFormat


@dataclass
class BatchResult:
    done: list[BatchItem] = field(default_factory=list)
    skipped: list[BatchItem] = field(default_factory=list)
    failed: list[tuple[BatchItem, str]] = field(default_factory=list)

    def summary(self) -> str:
        return f"{len(self.done)} colorized, {len(self.skipped)} skipped, {len(self.failed)} failed"


def run_batch(
    items: Sequence[BatchItem],
    get_model: Callable[[], ColorizerModel],
    params: Mapping[str, Any],
    post: Postprocess,
    *,
    quality: int = 95,
    overwrite: bool = False,
    ctx: JobContext | None = None,
) -> BatchResult:
    """Colorize ``items``, skipping existing outputs unless ``overwrite``.

    The model is loaded (``get_model``) only if something needs doing; a load failure
    propagates. Per-file errors are collected, not raised. Cancellation via ``ctx`` raises
    ``Cancelled`` between files.
    """
    result = BatchResult()
    pending: list[BatchItem] = []
    for item in items:
        if item.dst.exists() and not overwrite:
            log.info("skip %s (exists)", item.dst)
            result.skipped.append(item)
        else:
            pending.append(item)
    if not pending:
        return result
    if ctx:
        ctx.progress(0.0, "Loading model")
    model = get_model()
    n = len(pending)
    for i, item in enumerate(pending, 1):
        if ctx:
            ctx.progress((i - 1) / n, f"{i}/{n}: {item.src.name}")
        try:
            colorize_file(item.src, item.dst, model, params, post, fmt=item.fmt, quality=quality)
        except Exception as e:
            log.error("[%d/%d] FAILED %s: %s", i, n, item.src, e)
            log.debug("traceback", exc_info=True)
            result.failed.append((item, str(e)))
        else:
            log.info("[%d/%d] %s -> %s", i, n, item.src, item.dst)
            result.done.append(item)
    return result
