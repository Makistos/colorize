"""Interactive-use caches: decoded images and inferred ab, so postprocess re-renders are cheap.

Used by the UI. Holds no Gradio state; safe to call from multiple threads.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

from colorizer.core.pipeline import (
    L_to_gray,
    SourceImage,
    gray_to_L,
    infer_ab,
    load_image,
)
from colorizer.core.registry import Registry
from colorizer.core.restore import RestoreStep, run_restorer, validate_restore_params
from colorizer.core.runtime import Device
from colorizer.core.worker import JobContext

K = TypeVar("K")
V = TypeVar("V")


class LRU(Generic[K, V]):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


@dataclass(frozen=True, eq=False)
class LoadedImage:
    src: SourceImage
    digest: str  # content hash of L (or of the restore chain); identifies it in cache keys


def ab_key(image_digest: str, model_id: str, params: Mapping[str, Any]) -> str:
    """Stable hash of image + model + validated params."""
    blob = json.dumps([image_digest, model_id, params], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class Session:
    def __init__(
        self, registry: Registry, max_images: int = 4, max_ab: int = 32, max_restored: int = 4
    ) -> None:
        self.registry = registry
        self._images: LRU[tuple[str, int, int], LoadedImage] = LRU(max_images)
        self._ab: LRU[str, np.ndarray] = LRU(max_ab)
        # Restored images can be large (upscaled), hence a small default.
        self._restored: LRU[str, LoadedImage] = LRU(max_restored)
        self.inference_count = 0  # number of model calls; lets tests prove cache hits
        self.restore_count = 0  # number of restorer calls

    def load(self, path: Path) -> LoadedImage:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        cached = self._images.get(key)
        if cached is not None:
            return cached
        src = load_image(path)
        loaded = LoadedImage(src, hashlib.sha256(src.L.tobytes()).hexdigest())
        self._images.put(key, loaded)
        return loaded

    def _restore_keys(self, image: LoadedImage, steps: Sequence[RestoreStep]) -> list[str]:
        """Cache key after each step (each depends on all before it). Validates params."""
        keys, prev = [], image.digest
        for step in steps:
            cls = self.registry.restorers.get_class(step.id)
            prev = ab_key(prev, f"restore:{step.id}", validate_restore_params(cls, step.params))
            keys.append(prev)
        return keys

    def cached_restored(
        self, image: LoadedImage, steps: Sequence[RestoreStep]
    ) -> LoadedImage | None:
        """The result of a previous ``get_restored`` with the same inputs, or ``None``."""
        if not steps:
            return image
        return self._restored.get(self._restore_keys(image, steps)[-1])

    def get_restored(
        self,
        image: LoadedImage,
        steps: Sequence[RestoreStep],
        device: Device,
        ctx: JobContext | None = None,
    ) -> LoadedImage:
        """Run the restore chain, reusing the longest cached prefix. Raises ``ValueError``
        for bad params before loading anything. The result's digest is the chain key, so
        ab cached for it is invalidated when any restore param changes."""
        keys = self._restore_keys(image, steps)
        current, start = image, 0
        for i in range(len(steps), 0, -1):
            cached = self._restored.get(keys[i - 1])
            if cached is not None:
                current, start = cached, i
                break
        n = len(steps)
        for i in range(start, n):
            step = steps[i]
            if ctx:
                ctx.progress(0.3 * i / n, f"Loading {step.id}")
            restorer = self.registry.restorers.get(step.id, device)
            if ctx:
                ctx.progress(0.3 * (i + 0.2) / n, f"Restoring: {step.id}")

                def on_step(fraction: float, i: int = i, name: str = step.id) -> None:
                    ctx.progress(0.3 * (i + 0.2 + 0.8 * fraction) / n, f"Restoring: {name}")

                restorer.step_callback = on_step
            try:
                gray = run_restorer(restorer, L_to_gray(current.src.L), step.params)
            finally:
                restorer.step_callback = None
            self.restore_count += 1
            src = SourceImage(gray_to_L(gray), current.src.bit_depth, current.src.exif)
            current = LoadedImage(src, keys[i])
            self._restored.put(keys[i], current)
        return current

    def _key(self, image: LoadedImage, model_id: str, params: Mapping[str, Any]) -> str:
        validated = self.registry.get_class(model_id).validate_params(params)
        return ab_key(image.digest, model_id, validated)

    def cached_ab(
        self, image: LoadedImage, model_id: str, params: Mapping[str, Any]
    ) -> np.ndarray | None:
        """The ab from a previous ``get_ab`` with the same inputs, or ``None``."""
        return self._ab.get(self._key(image, model_id, params))

    def get_ab(
        self,
        image: LoadedImage,
        model_id: str,
        params: Mapping[str, Any],
        device: Device,
        ctx: JobContext | None = None,
    ) -> np.ndarray:
        """Return ab, running the model only on a cache miss. Raises ``ValueError`` for bad
        params before loading anything."""
        key = self._key(image, model_id, params)
        cached = self._ab.get(key)
        if cached is not None:
            return cached
        if ctx:
            ctx.progress(0.1, f"Loading {model_id}")
        model = self.registry.get(model_id, device)
        if ctx:
            ctx.progress(0.3, f"Running {model_id}")

            def on_step(fraction: float) -> None:
                ctx.progress(0.3 + 0.6 * fraction, f"Running {model_id}")

            model.step_callback = on_step
        try:
            ab = infer_ab(image.src, model, params)
        finally:
            model.step_callback = None
        self.inference_count += 1
        self._ab.put(key, ab)
        return ab
