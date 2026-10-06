"""Export DDColor (large, tiny, artistic) to ONNX.

    uv run --group export python tools/export_onnx/ddcolor.py [--variant large|tiny|artistic|all]

Fetches the upstream repo at a pinned commit (export-time only; never imported by the app)
and builds the network with its own ``ddcolor.DDColor`` class, loading the official weights
with ``strict=True``. Spectral norms are folded into plain weights before export.

ONNX graph: input ``gray`` (1x3xHxW, neutral gray RGB in [0,1], H and W multiples of 32,
dynamic) -> output ``ab`` (1x2xHxW, Lab ab).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import remove_spectral_norm

from colorizer.core import weights
from colorizer.models._onnx import onnx_path
from colorizer.models.ddcolor import ENCODERS, UPSTREAM_COMMIT, UPSTREAM_REPO, VARIANTS


def upstream_checkout() -> Path:
    """Shallow-fetch the pinned upstream commit into the cache (the commit hash pins it)."""
    root = weights.cache_dir() / "src" / f"DDColor-{UPSTREAM_COMMIT[:12]}"
    if not (root / "ddcolor" / "model.py").exists():
        root.mkdir(parents=True, exist_ok=True)
        for cmd in (
            ["git", "init", "-q"],
            ["git", "fetch", "-q", "--depth", "1", UPSTREAM_REPO, UPSTREAM_COMMIT],
            ["git", "checkout", "-q", "FETCH_HEAD"],
        ):
            subprocess.run(cmd, cwd=root, check=True)
    return root


def build(variant: str) -> nn.Module:
    sys.path.insert(0, str(upstream_checkout()))
    from ddcolor import DDColor  # type: ignore[import-not-found]

    encoder = "convnext-t" if ENCODERS[variant] == "tiny" else "convnext-l"
    # Same constructor arguments as upstream ddcolor.pipeline.build_ddcolor_model.
    net = DDColor(
        encoder_name=encoder,
        decoder_name="MultiScaleColorDecoder",
        input_size=[512, 512],
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    )
    ckpt = torch.load(
        weights.ensure(VARIANTS[variant].checkpoint), map_location="cpu", weights_only=True
    )
    state = ckpt.get("params", ckpt)
    net.load_state_dict(state, strict=True)
    net.eval()
    for m in net.modules():
        if hasattr(m, "weight_orig"):
            remove_spectral_norm(m)
    return net


def export(variant: str, out: Path) -> None:
    net = build(variant)
    dummy = torch.rand(1, 3, 512, 512)
    out.parent.mkdir(parents=True, exist_ok=True)
    # In units of 32 px: 64..2048 px.
    h, w = torch.export.Dim("h", min=2, max=64), torch.export.Dim("w", min=2, max=64)
    torch.onnx.export(
        net,
        (dummy,),
        str(out),
        input_names=["gray"],
        output_names=["ab"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"x": {2: 32 * h, 3: 32 * w}},
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    err = 0.0
    for shape in ((512, 512), (256, 384)):
        x = torch.rand(1, 3, *shape)
        with torch.no_grad():
            expected = net(x).numpy()
        (got,) = sess.run(None, {"gray": x.numpy()})
        err = max(err, float(np.abs(got - expected).max()))
    if err > 1e-2:
        raise SystemExit(f"{variant}: ONNX differs from PyTorch (max abs err {err:.4g})")
    logging.info("wrote %s (max abs err %.2e)", out, err)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = parser.parse_args()
    for variant in VARIANTS if args.variant == "all" else [args.variant]:
        export(variant, onnx_path(VARIANTS[variant].onnx))


if __name__ == "__main__":
    main()
