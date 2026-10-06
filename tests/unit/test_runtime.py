import logging
from pathlib import Path

import pytest

from colorizer.core import runtime
from colorizer.core.runtime import CPU, Device, create_session, select_device

ALL = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def providers(monkeypatch):
    def set_(names):
        monkeypatch.setattr(runtime.ort, "get_available_providers", lambda: names)

    monkeypatch.delenv(runtime.ENV_DEVICE, raising=False)
    return set_


def test_auto_picks_highest_priority(providers):
    providers(["CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider"])
    assert select_device().name == "cuda"
    providers(["CoreMLExecutionProvider", "CPUExecutionProvider"])
    assert select_device("auto").name == "coreml"
    providers(["CPUExecutionProvider"])
    assert select_device() == CPU


def test_env_override(providers, monkeypatch):
    providers(ALL)
    monkeypatch.setenv(runtime.ENV_DEVICE, "cpu")
    assert select_device() == CPU
    assert select_device("directml").name == "directml"  # explicit beats env


@pytest.mark.parametrize("name", ["tpu", "rocm"])
def test_unknown_or_unavailable(providers, name):
    providers(ALL)
    with pytest.raises(ValueError):
        select_device(name)


class FakeSession:
    def __init__(self, path, sess_options=None, providers=()):
        if providers[0] == "CUDAExecutionProvider" and FakeSession.cuda_mode == "raise":
            raise RuntimeError("CUDA driver too old")
        self.providers = list(providers)
        if FakeSession.cuda_mode == "drop":
            self.providers = [p for p in self.providers if p != "CUDAExecutionProvider"]

    def get_providers(self):
        return self.providers


@pytest.mark.parametrize("mode", ["raise", "drop"])
def test_gpu_failure_falls_back_to_cpu(monkeypatch, caplog, mode):
    monkeypatch.setattr(runtime.ort, "InferenceSession", FakeSession)
    FakeSession.cuda_mode = mode
    with caplog.at_level(logging.WARNING):
        session, dev = create_session(Path("m.onnx"), Device("cuda", "CUDAExecutionProvider"))
    assert dev == CPU and session.get_providers() == ["CPUExecutionProvider"]
    assert "falling back to CPU" in caplog.text


def test_gpu_success(monkeypatch):
    monkeypatch.setattr(runtime.ort, "InferenceSession", FakeSession)
    FakeSession.cuda_mode = "ok"
    cuda = Device("cuda", "CUDAExecutionProvider")
    _, dev = create_session(Path("m.onnx"), cuda)
    assert dev == cuda


def test_cuda_preloads_pip_libraries_once(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(runtime.ort, "preload_dlls", lambda: calls.append(1), raising=False)
    monkeypatch.setattr(runtime, "_cuda_preloaded", False)
    FakeSession.cuda_mode = "ok"
    cuda = Device("cuda", "CUDAExecutionProvider")
    create_session(Path("a.onnx"), cuda)
    create_session(Path("b.onnx"), cuda)
    create_session(Path("c.onnx"), Device("directml", "DmlExecutionProvider"))
    assert calls == [1]


def test_webgpu_ranks_below_native_gpu_providers_above_cpu(providers):
    providers(["CPUExecutionProvider", "WebGpuExecutionProvider"])
    assert select_device().name == "webgpu"
    providers(["WebGpuExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    assert select_device().name == "cuda"
