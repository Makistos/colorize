import hashlib
import http.server
import threading
from functools import partial

import pytest

from colorizer.core import weights
from colorizer.core.weights import WeightFile

PAYLOAD = b"fake weights " * 1000
SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    root.mkdir()
    (root / "w.bin").write_bytes(PAYLOAD)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_download_and_cache_hit(server, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    w = WeightFile("w.bin", f"{server}/w.bin", SHA)
    path = weights.ensure(w, cache)
    assert path.read_bytes() == PAYLOAD
    # A second call must not touch the network.
    monkeypatch.setattr(weights.urllib.request, "urlopen", lambda *a: pytest.fail("network"))
    assert weights.ensure(w, cache) == path


def test_checksum_mismatch(server, tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="checksum"):
        weights.ensure(WeightFile("w.bin", f"{server}/w.bin", "0" * 64), cache)
    assert list(cache.iterdir()) == []


def test_partial_download_not_cached(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "w.bin.part").write_bytes(PAYLOAD[:10])
    with pytest.raises(OSError):
        weights.ensure(WeightFile("w.bin", "http://127.0.0.1:9/w.bin", SHA), cache)
    assert not (cache / "w.bin").exists()


def test_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv(weights.ENV_CACHE_DIR, str(tmp_path))
    assert weights.cache_dir() == tmp_path
