"""load → Lab split → infer → recombine → postprocess → save."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.color import lab2rgb, rgb2lab

from colorizer.core import postprocess
from colorizer.core.base import ColorizerModel
from colorizer.core.postprocess import Postprocess

log = logging.getLogger(__name__)

OutputFormat = Literal["png", "jpg", "tiff"]
FORMATS: dict[str, str] = {"png": "PNG", "jpg": "JPEG", "tiff": "TIFF"}
EXTENSIONS: dict[str, str] = {"png": ".png", "jpg": ".jpg", "tiff": ".tiff"}
# Input file suffixes considered images when scanning directories.
INPUT_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})

_MODES_16BIT = frozenset({"I;16", "I;16L", "I;16B", "I;16N", "I"})
_ORIENTATION_TAG = 0x0112


@dataclass(frozen=True, eq=False)
class SourceImage:
    """A grayscale source: full-resolution L plus what is needed to save the result."""

    L: np.ndarray  # float32 HxW in [0, 100]
    bit_depth: int = 8  # 8 or 16
    exif: bytes | None = None

    @property
    def size(self) -> tuple[int, int]:
        """(height, width)"""
        h, w = self.L.shape
        return h, w

    @classmethod
    def from_rgb(
        cls, rgb: np.ndarray, bit_depth: int = 8, exif: bytes | None = None
    ) -> SourceImage:
        """Build from an RGB (HxWx3) or gray (HxW) array, float in [0,1] or uint8/uint16."""
        arr = _to_unit_float(rgb)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected HxW or HxWx3 image, got shape {rgb.shape}")
        L = rgb2lab(arr)[..., 0].astype(np.float32)
        return cls(L=L, bit_depth=bit_depth, exif=exif)


def _to_unit_float(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float64) / 65535.0
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float64), 0.0, 1.0)
    raise ValueError(f"unsupported image dtype {arr.dtype}")


def load_image(path: Path) -> SourceImage:
    """Open an image, apply EXIF orientation and reduce it to its L channel.

    Colour inputs (e.g. sepia scans) are deliberately discarded down to lightness.
    """
    with Image.open(path) as opened:
        im = ImageOps.exif_transpose(opened)
    exif = im.getexif()
    if _ORIENTATION_TAG in exif:
        exif[_ORIENTATION_TAG] = 1  # pixels are already rotated
    exif_bytes = exif.tobytes() if len(exif) else None

    if im.mode in _MODES_16BIT:
        gray = np.asarray(im, dtype=np.float64) / 65535.0
        return SourceImage.from_rgb(np.clip(gray, 0.0, 1.0), bit_depth=16, exif=exif_bytes)
    rgb = np.asarray(im.convert("RGB"))
    return SourceImage.from_rgb(rgb, bit_depth=8, exif=exif_bytes)


def infer_ab(src: SourceImage, model: ColorizerModel, params: Mapping[str, Any]) -> np.ndarray:
    """Validate ``params`` and run the model. Returns ab at the model's working size."""
    validated = model.validate_params(params)
    ab = model.predict_ab(src.L, **validated)
    if ab.ndim != 3 or ab.shape[2] != 2:
        raise ValueError(f"{model.id}.predict_ab returned shape {ab.shape}, expected h x w x 2")
    return ab.astype(np.float32, copy=False)


def render(src: SourceImage, ab: np.ndarray, post: Postprocess | None = None) -> np.ndarray:
    """Upscale ``ab`` to full resolution, postprocess, recombine with L.

    Returns RGB float64 HxWx3 in [0, 1]. Cheap relative to inference, so callers can
    cache ``ab`` and re-render when only postprocess settings change.
    """
    h, w = src.size
    if ab.shape[:2] != (h, w):
        ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)
    ab = postprocess.apply(ab, post or Postprocess())
    lab = np.dstack([src.L.astype(np.float64), ab.astype(np.float64)])
    with warnings.catch_warnings():
        # Out-of-gamut colours are clipped; skimage warns about each one.
        warnings.simplefilter("ignore", UserWarning)
        rgb = np.asarray(lab2rgb(lab), dtype=np.float64)
    return np.clip(rgb, 0.0, 1.0)


def save_image(
    rgb: np.ndarray,
    path: Path,
    fmt: OutputFormat = "png",
    bit_depth: int = 8,
    exif: bytes | None = None,
    quality: int = 95,
) -> None:
    """Save RGB float [0,1]. PNG/TIFF are 16-bit when ``bit_depth`` is 16; JPEG is always 8-bit."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {list(FORMATS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if bit_depth == 16 and fmt != "jpg":
        # Pillow cannot write 48-bit RGB; OpenCV can, but drops EXIF.
        data = np.round(rgb * 65535.0).astype(np.uint16)
        ok, buf = cv2.imencode(EXTENSIONS[fmt], cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
        if not ok:
            raise OSError(f"failed to encode {path}")
        path.write_bytes(buf.tobytes())  # imwrite can't handle non-ASCII paths on Windows
        if exif:
            log.warning("%s: EXIF is not preserved in 16-bit output", path.name)
        return
    im = Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")
    kwargs: dict[str, Any] = {}
    if exif:
        kwargs["exif"] = exif
    if fmt == "jpg":
        kwargs.update(quality=quality, subsampling=0)
    im.save(path, format=FORMATS[fmt], **kwargs)


def colorize_file(
    src_path: Path,
    dst_path: Path,
    model: ColorizerModel,
    params: Mapping[str, Any],
    post: Postprocess | None = None,
    fmt: OutputFormat = "png",
    quality: int = 95,
) -> None:
    src = load_image(src_path)
    ab = infer_ab(src, model, params)
    rgb = render(src, ab, post)
    save_image(rgb, dst_path, fmt=fmt, bit_depth=src.bit_depth, exif=src.exif, quality=quality)
