"""load → [restore …] → Lab split → infer → recombine → postprocess → save."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from colorizer.core import postprocess
from colorizer.core.base import ColorizerModel
from colorizer.core.postprocess import Postprocess
from colorizer.core.restore import Restorer, run_restorer, validate_restore_params

log = logging.getLogger(__name__)

OutputFormat = Literal["png", "jpg", "tiff"]
FORMATS: dict[str, str] = {"png": "PNG", "jpg": "JPEG", "tiff": "TIFF"}
EXTENSIONS: dict[str, str] = {"png": ".png", "jpg": ".jpg", "tiff": ".tiff"}
# Output file suffix -> format.
SUFFIX_FORMATS: dict[str, OutputFormat] = {
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpg",
    ".tif": "tiff",
    ".tiff": "tiff",
}
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


def L_to_gray(L: np.ndarray) -> np.ndarray:
    """L in [0, 100] -> the sRGB value (float32 in [0, 1]) of the neutral gray with that L."""
    lin = _finv((L.astype(np.float64) + 16.0) / 116.0)
    gray = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.maximum(lin, 0) ** (1 / 2.4) - 0.055)
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def gray_to_L(gray: np.ndarray) -> np.ndarray:
    """Inverse of ``L_to_gray``: sRGB gray in [0, 1] -> L (float32 in [0, 100])."""
    g = np.clip(gray.astype(np.float64), 0.0, 1.0)
    Y = np.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4)
    f = np.where(Y > (6 / 29) ** 3, np.cbrt(Y), Y / (3 * (6 / 29) ** 2) + 4 / 29)
    return (116.0 * f - 16.0).astype(np.float32)


def restore_image(
    src: SourceImage, steps: Sequence[tuple[Restorer, Mapping[str, Any]]]
) -> SourceImage:
    """Run restorers in order on the grayscale image. All params are validated first."""
    for restorer, params in steps:
        validate_restore_params(type(restorer), params)
    if not steps:
        return src
    gray = L_to_gray(src.L)
    for restorer, params in steps:
        gray = run_restorer(restorer, gray, params)
    return SourceImage(L=gray_to_L(gray), bit_depth=src.bit_depth, exif=src.exif)


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
    return lab_to_rgb(src.L, ab)


# CIE Lab (D65, 2°) -> linear sRGB, with the same constants as skimage's rgb2lab.
_WHITE_X, _WHITE_Z = 0.95047, 1.08883
_RGB_FROM_XYZ = (
    (3.24048134, -1.53715152, -0.49853633),
    (-0.96925495, 1.87599, 0.04155593),
    (0.05564664, -0.20404134, 1.05731107),
)
_GAMUT_EPS = 1e-6
_GAMUT_STEPS = 12  # binary-search steps on the chroma scale (precision 2**-12)


def _finv(t: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.where(t > 6 / 29, t * t * t, (3 * (6 / 29) ** 2) * (t - 4 / 29))
    return out


def _lab_to_linear(fy: np.ndarray, Y: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel ``fy = (L+16)/116``, ``Y = finv(fy)``, a, b (all (N,)) -> linear sRGB (3, N)."""
    X = _WHITE_X * _finv(fy + a / 500.0)
    Z = _WHITE_Z * _finv(fy - b / 200.0)
    return np.stack([r0 * X + r1 * Y + r2 * Z for r0, r1, r2 in _RGB_FROM_XYZ])


def _in_gamut(lin: np.ndarray) -> np.ndarray:
    ok: np.ndarray = ((lin >= -_GAMUT_EPS) & (lin <= 1.0 + _GAMUT_EPS)).all(axis=0)
    return ok


def lab_to_rgb(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Lab -> sRGB float64 HxWx3 in [0, 1] that keeps L exact.

    Out-of-gamut pixels lose chroma at constant hue (binary search on an ab scale factor)
    instead of having RGB clipped per channel, which would change their lightness.
    """
    h, w = L.shape
    fy = (L.reshape(-1).astype(np.float64) + 16.0) / 116.0
    Y = _finv(fy)
    a = ab[..., 0].reshape(-1).astype(np.float64)
    b = ab[..., 1].reshape(-1).astype(np.float64)
    lin = _lab_to_linear(fy, Y, a, b)
    out = np.flatnonzero(~_in_gamut(lin))
    if out.size:
        fy_o, Y_o, a_o, b_o = fy[out], Y[out], a[out], b[out]
        # float32 is plenty for the search (error ~1e-7 < _GAMUT_EPS) and twice as fast.
        f32 = [x.astype(np.float32) for x in (fy_o, Y_o, a_o, b_o)]
        lo, hi = np.zeros(out.size, np.float32), np.ones(out.size, np.float32)
        for _ in range(_GAMUT_STEPS):
            mid = (lo + hi) * np.float32(0.5)
            ok = _in_gamut(_lab_to_linear(f32[0], f32[1], f32[2] * mid, f32[3] * mid))
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        k = lo.astype(np.float64)
        lin[:, out] = _lab_to_linear(fy_o, Y_o, a_o * k, b_o * k)
    lin = np.clip(lin, 0.0, 1.0)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    out_rgb: np.ndarray = np.clip(rgb.T, 0.0, 1.0).reshape(h, w, 3)
    return out_rgb


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
    restore: Sequence[tuple[Restorer, Mapping[str, Any]]] = (),
) -> None:
    src = restore_image(load_image(src_path), restore)
    ab = infer_ab(src, model, params)
    rgb = render(src, ab, post)
    save_image(rgb, dst_path, fmt=fmt, bit_depth=src.bit_depth, exif=src.exif, quality=quality)
