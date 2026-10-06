"""Device and ONNX Runtime execution-provider selection."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort

log = logging.getLogger(__name__)

ENV_DEVICE = "COLORIZER_DEVICE"
CPU_PROVIDER = "CPUExecutionProvider"

# Priority order for automatic selection: user-facing name -> ONNX Runtime provider.
PROVIDERS: dict[str, str] = {
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": CPU_PROVIDER,
}


@dataclass(frozen=True)
class Device:
    name: str  # key of PROVIDERS
    provider: str

    @property
    def is_gpu(self) -> bool:
        return self.provider != CPU_PROVIDER


CPU = Device("cpu", CPU_PROVIDER)


def available_devices() -> list[Device]:
    """Devices whose provider is available in this onnxruntime build, in priority order."""
    available = set(ort.get_available_providers())
    return [Device(name, prov) for name, prov in PROVIDERS.items() if prov in available]


def select_device(requested: str | None = None) -> Device:
    """Pick a device: explicit request, then ``$COLORIZER_DEVICE``, then the best available.

    ``"auto"`` or ``None`` means automatic. Raises ``ValueError`` for an unknown name or a
    device that this onnxruntime build does not provide.
    """
    name = (requested or os.environ.get(ENV_DEVICE) or "auto").strip().lower()
    devices = available_devices()
    if name == "auto":
        return devices[0] if devices else CPU
    if name not in PROVIDERS:
        raise ValueError(f"unknown device {name!r}; expected auto or one of {list(PROVIDERS)}")
    for dev in devices:
        if dev.name == name:
            return dev
    raise ValueError(f"device {name!r} is not available; available: {[d.name for d in devices]}")


def create_session(model_path: Path, device: Device) -> tuple[Any, Device]:
    """Create an ``InferenceSession`` on ``device``, falling back to CPU with a warning.

    Returns the session and the device it actually runs on.
    """
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # errors only; we report fallbacks ourselves
    if device.is_gpu:
        try:
            session = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=[device.provider, CPU_PROVIDER]
            )
        except Exception as e:  # provider DLLs missing, driver errors, OOM, ...
            log.warning(
                "%s failed to load %s (%s); falling back to CPU", device.name, model_path.name, e
            )
        else:
            # onnxruntime silently drops providers it cannot initialise.
            if session.get_providers()[0] == device.provider:
                return session, device
            log.warning(
                "%s is unavailable for %s; falling back to CPU", device.name, model_path.name
            )
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=[CPU_PROVIDER])
    return session, CPU
