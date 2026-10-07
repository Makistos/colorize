"""Stable Diffusion 1.5 + "brightness" ControlNet recolouring (optional extra).

The ControlNet is conditioned on a grayscale image and trained to restore its colours, so
the generated image keeps the photo's structure; only its ab channels are used.

Install: ``uv sync --extra <runtime> --extra diffusion`` (NVIDIA/Apple/CPU) or
``--extra diffusion-rocm`` (AMD on Linux). Weights download from Hugging Face on first use
(about 3.6 GB) into ``<cache dir>/huggingface``.

Sources, pinned by commit (content-addressed):
- https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M),
  fp16 weights; the safety checker is not loaded.
- https://huggingface.co/latentcat/control_v1p_sd15_brightness (CreativeML OpenRAIL-M)
"""

from __future__ import annotations

import logging
import random
from importlib.util import find_spec
from typing import Any

import cv2
import numpy as np
from skimage.color import rgb2lab

from colorizer.core.base import ColorizerModel
from colorizer.core.params import Param
from colorizer.core.runtime import Device
from colorizer.core.weights import cache_dir
from colorizer.models._onnx import gray_rgb

log = logging.getLogger(__name__)

SD_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
CONTROLNET_REPO = "latentcat/control_v1p_sd15_brightness"
CONTROLNET_REVISION = "8509361eb1ba89c03839040ed8c75e5f11bbd9c5"
WORKING_SIDE = 512  # SD 1.5's native resolution (long side)
INSTALL_HINT = (
    "uv sync --extra <runtime> --extra diffusion   (AMD on Linux: --extra diffusion-rocm)"
)


def torch_device(device: Device) -> str:
    """PyTorch device for an ONNX-style device choice. ROCm builds report as "cuda"."""
    import torch

    if device.name == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    log.warning("no GPU available to PyTorch; Stable Diffusion runs on CPU (slow)")
    return "cpu"


def working_size(h: int, w: int) -> tuple[int, int]:
    """Long side WORKING_SIDE, both sides multiples of 64 (keeps the aspect ratio)."""
    scale = WORKING_SIDE / max(h, w)
    return max(64, round(h * scale / 64) * 64), max(64, round(w * scale / 64) * 64)


class SDControlNet(ColorizerModel):
    id = "sd_controlnet"
    display_name = "Stable Diffusion + ControlNet"
    license = "CreativeML OpenRAIL-M (SD 1.5; latentcat brightness ControlNet)"
    params = (
        Param(
            "prompt",
            "str",
            "a colorized vintage photograph, natural realistic colors",
            help="Describe the scene and colours you want.",
        ),
        Param(
            "negative_prompt",
            "str",
            "oversaturated, cartoon, painting, illustration, monochrome, sepia, blurry",
            help="What to avoid.",
        ),
        Param("steps", "int", 20, min=10, max=50, help="Denoising steps; more is slower."),
        Param(
            "cfg",
            "float",
            7.0,
            min=1.0,
            max=15.0,
            step=0.5,
            help="How strongly to follow the prompt.",
        ),
        Param(
            "strength",
            "float",
            1.0,
            min=0.0,
            max=1.0,
            step=0.05,
            help="How far the colours may move from the gray input (1 = freely).",
        ),
        Param("seed", "seed", -1, min=-1, help="-1 = random. Change it for another variation."),
    )

    def __init__(self) -> None:
        self._pipe: Any = None
        self._device = "cpu"

    @classmethod
    def available(cls) -> bool:
        return find_spec("torch") is not None and find_spec("diffusers") is not None

    def load(self, device: Device) -> None:
        try:
            import torch
            from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        except ImportError as e:
            raise ImportError(f"Stable Diffusion needs the diffusion extra: {INSTALL_HINT}") from e
        self._device = torch_device(device)
        dtype = torch.float32 if self._device == "cpu" else torch.float16
        hf_cache = str(cache_dir() / "huggingface")
        log.info("loading Stable Diffusion on %s (first use downloads ~3.6 GB)", self._device)
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_REPO, revision=CONTROLNET_REVISION, torch_dtype=dtype, cache_dir=hf_cache
        )
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            SD_REPO,
            revision=SD_REVISION,
            controlnet=controlnet,
            variant="fp16",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=hf_cache,
        )
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe.to(self._device)

    def unload(self) -> None:
        self._pipe = None
        if self._device == "cuda":
            import torch

            torch.cuda.empty_cache()

    def predict_ab(self, L: np.ndarray, **params: Any) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("model not loaded")
        import torch
        from PIL import Image

        h, w = working_size(*L.shape)
        steps, strength = int(params["steps"]), float(params["strength"])
        if strength * steps < 1:  # nothing would be denoised: the input stays gray
            return np.zeros((h, w, 2), np.float32)
        gray = gray_rgb(cv2.resize(L, (w, h), interpolation=cv2.INTER_AREA))
        gray_img = Image.fromarray(np.round(gray * 255).astype(np.uint8))
        seed = int(params["seed"])
        generator = torch.Generator("cpu").manual_seed(
            seed if seed >= 0 else random.randrange(2**32)
        )
        total = max(1, int(steps * strength))
        callback = self.step_callback

        def on_step_end(pipe: Any, step: int, timestep: Any, kwargs: dict[str, Any]) -> Any:
            if callback:
                callback((step + 1) / total)  # raises Cancelled to stop mid-generation
            return kwargs

        result = self._pipe(
            params["prompt"],
            negative_prompt=params["negative_prompt"] or None,
            image=gray_img,
            control_image=gray_img,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=float(params["cfg"]),
            generator=generator,
            callback_on_step_end=on_step_end,
            output_type="np",
        ).images[0]
        lab = rgb2lab(np.clip(result, 0.0, 1.0).astype(np.float64))
        return np.ascontiguousarray(lab[..., 1:], dtype=np.float32)
