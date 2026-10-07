"""Interactive-use caches: decoded images and inferred ab, so postprocess re-renders are cheap.

Used by the UI. Holds no Gradio state; safe to call from multiple threads.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

from colorizer.core.pipeline import SourceImage, infer_ab, load_image
from colorizer.core.registry import Registry
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
    digest: str  # content hash of L; identifies the image in ab cache keys


def ab_key(image_digest: str, model_id: str, params: Mapping[str, Any]) -> str:
    """Stable hash of image + model + validated params."""
    blob = json.dumps([image_digest, model_id, params], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class Session:
    def __init__(self, registry: Registry, max_images: int = 4, max_ab: int = 32) -> None:
        self.registry = registry
        self._images: LRU[tuple[str, int, int], LoadedImage] = LRU(max_images)
        self._ab: LRU[str, np.ndarray] = LRU(max_ab)
        self.inference_count = 0  # number of model calls; lets tests prove cache hits

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
