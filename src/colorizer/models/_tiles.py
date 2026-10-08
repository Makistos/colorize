"""Tiled inference for image-to-image networks, so large images fit in memory."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

TILE_PAD = 10  # context pixels around each tile, as upstream Real-ESRGAN


def run_tiled(
    run: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    scale: int,
    tile: int,
    pad: int = TILE_PAD,
    on_tile: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Apply ``run`` (1xCxHxW -> 1xC'x(H*scale)x(W*scale)) to ``x`` in ``tile``-sized pieces.

    Each piece gets ``pad`` pixels of surrounding context, which is cropped from the output,
    so seams are hidden. ``tile`` <= 0 runs the whole image at once.
    """
    _, _, h, w = x.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return run(x)
    out: np.ndarray | None = None
    ys, xs = range(0, h, tile), range(0, w, tile)
    total, done = len(ys) * len(xs), 0
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
            py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
            piece = run(np.ascontiguousarray(x[:, :, py0:py1, px0:px1]))
            if out is None:
                out = np.empty((1, piece.shape[1], h * scale, w * scale), piece.dtype)
            oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
            out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = piece[
                :, :, oy : oy + (y1 - y0) * scale, ox : ox + (x1 - x0) * scale
            ]
            done += 1
            if on_tile:
                on_tile(done / total)
    assert out is not None
    return out
