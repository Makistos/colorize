"""DeOldify NoGAN generators (artistic, stable).

Source: https://github.com/jantic/DeOldify (MIT). Weights are the official release files
linked from the upstream README. ONNX files are produced by ``tools/export_onnx/deoldify.py``,
which rebuilds the network without fastai.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.color import rgb2lab

from colorizer.core.params import Param
from colorizer.core.weights import WeightFile
from colorizer.models._onnx import (
    OnnxVariant,
    OnnxVariantModel,
    gray_rgb,
    onnx_path,
    resize_L,
    to_nchw,
)

__all__ = ["VARIANTS", "DeOldify", "onnx_path"]

RENDER_BASE = 16  # working size = render_factor * 16, as upstream

VARIANTS: dict[str, OnnxVariant] = {
    "artistic": OnnxVariant(
        WeightFile(
            "ColorizeArtistic_gen.pth",
            "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth",
            "3f750246fa220529323b85a8905f9b49c0e5d427099185334d048fb5b5e22477",
        ),
        "deoldify_artistic.onnx",
    ),
    "stable": OnnxVariant(
        WeightFile(
            "ColorizeStable_gen.pth",
            "https://www.dropbox.com/s/axsd2g85uyixaho/ColorizeStable_gen.pth?dl=1",
            "ca9cd7f43fb8b222c9a70f7b292e305a000694b0ff9d2ae4a6747b1a2e1ee5af",
        ),
        "deoldify_stable.onnx",
    ),
}


class DeOldify(OnnxVariantModel):
    id = "deoldify"
    display_name = "DeOldify"
    license = "MIT (jantic/DeOldify)"
    variants = VARIANTS
    export_hint = "uv run --group export python tools/export_onnx/deoldify.py"
    params = (
        Param(
            "variant",
            "choice",
            "artistic",
            choices=("artistic", "stable"),
            help="artistic: vivid, more detail; stable: fewer glitches on portraits/landscapes.",
        ),
        Param(
            "render_factor",
            "int",
            35,
            min=7,
            max=45,
            help="Working size = render_factor x 16 px. Lower is more vivid but blurrier.",
        ),
    )

    def predict_ab(self, L: np.ndarray, **params: Any) -> np.ndarray:
        size = params["render_factor"] * RENDER_BASE
        gray = gray_rgb(resize_L(L, size))
        (rgb,) = self.session(params["variant"]).run(None, {"gray": to_nchw(gray)})
        lab = rgb2lab(rgb[0].transpose(1, 2, 0).astype(np.float64))
        return np.ascontiguousarray(lab[..., 1:], dtype=np.float32)
