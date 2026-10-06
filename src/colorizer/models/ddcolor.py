"""DDColor: dual-decoder image colorization (Kang et al., ICCV 2023).

Source: https://github.com/piddnad/DDColor (Apache-2.0); weights from the authors'
Hugging Face repos, pinned by commit. ONNX files are produced by
``tools/export_onnx/ddcolor.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from colorizer.core.params import Param
from colorizer.core.weights import WeightFile
from colorizer.models._onnx import OnnxVariant, OnnxVariantModel, gray_rgb, resize_L, to_nchw

# Upstream code used only at export time.
UPSTREAM_REPO = "https://github.com/piddnad/DDColor.git"
UPSTREAM_COMMIT = "2adb63f2656ac41cbdf7b894cddd94121a3faf13"

_HF = "https://huggingface.co/piddnad/{repo}/resolve/{rev}/pytorch_model.bin"

VARIANTS: dict[str, OnnxVariant] = {
    # The authors recommend the ModelScope-trained large model for general use.
    "large": OnnxVariant(
        WeightFile(
            "ddcolor_modelscope.bin",
            _HF.format(repo="ddcolor_modelscope", rev="060f67494e31883a4b13cb27f889f3154847ada4"),
            "d81711971ec59200da26d5e8a1afae8dd3778d495ea8ad7a7dadc769f403f7e7",
        ),
        "ddcolor_large.onnx",
    ),
    "tiny": OnnxVariant(
        WeightFile(
            "ddcolor_paper_tiny.bin",
            _HF.format(repo="ddcolor_paper_tiny", rev="cf9fd99c1d7472689ec7413441c1b799a51866a3"),
            "8a1277bc90a1bfbb6d2d83933a9a6bc821931879ca93e26e4fcec12165d41fce",
        ),
        "ddcolor_tiny.onnx",
    ),
    "artistic": OnnxVariant(
        WeightFile(
            "ddcolor_artistic.bin",
            _HF.format(repo="ddcolor_artistic", rev="aa10f72fffc89a6658e37b48556050b4d9a26f63"),
            "5a076e84efce59848394f572a105f7b78f7f6e917f615d1ee88537241b96dca3",
        ),
        "ddcolor_artistic.onnx",
    ),
}
ENCODERS = {"large": "large", "tiny": "tiny", "artistic": "large"}


class DDColor(OnnxVariantModel):
    id = "ddcolor"
    display_name = "DDColor"
    license = "Apache-2.0 (piddnad/DDColor)"
    variants = VARIANTS
    export_hint = "uv run --group export python tools/export_onnx/ddcolor.py"
    params = (
        Param(
            "variant",
            "choice",
            "large",
            choices=("tiny", "large", "artistic"),
            help="large: best general quality; tiny: fast; artistic: more vivid.",
        ),
        Param(
            "size",
            "int",
            512,
            min=256,
            max=1024,
            step=64,
            help="Working resolution. Higher keeps small details but is slower.",
        ),
    )

    def predict_ab(self, L: np.ndarray, **params: Any) -> np.ndarray:
        gray = gray_rgb(resize_L(L, params["size"]))
        (ab,) = self.session(params["variant"]).run(None, {"gray": to_nchw(gray)})
        return np.ascontiguousarray(ab[0].transpose(1, 2, 0), dtype=np.float32)
