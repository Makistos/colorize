"""Download and cache model weights with SHA256 verification."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

ENV_CACHE_DIR = "COLORIZER_CACHE_DIR"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class WeightFile:
    filename: str
    url: str
    sha256: str


def cache_dir() -> Path:
    """``$COLORIZER_CACHE_DIR`` or ``~/.cache/colorizer``."""
    env = os.environ.get(ENV_CACHE_DIR)
    return Path(env) if env else Path.home() / ".cache" / "colorizer"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def ensure(weight: WeightFile, directory: Path | None = None) -> Path:
    """Return the local path of ``weight``, downloading and verifying it if needed.

    The download goes to a ``.part`` file that is renamed only after the checksum matches,
    so an interrupted download is never mistaken for a cached file.
    Raises ``ValueError`` on checksum mismatch and ``OSError`` on network errors.
    """
    directory = directory or cache_dir()
    target = directory / weight.filename
    if target.exists():
        return target
    directory.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    log.info("downloading %s from %s", weight.filename, weight.url)
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(weight.url) as resp, part.open("wb") as out:
            while chunk := resp.read(_CHUNK):
                h.update(chunk)
                out.write(chunk)
        digest = h.hexdigest()
        if digest != weight.sha256.lower():
            raise ValueError(
                f"checksum mismatch for {weight.filename}: expected {weight.sha256}, got {digest}"
            )
        shutil.move(part, target)
    finally:
        part.unlink(missing_ok=True)
    return target
