# Colorizer

Local-first AI colorization of black-and-white photographs. Pick a model, tweak its
parameters, and compare the results. Everything runs on your own machine.

- Models: Zhang ECCV16 and SIGGRAPH17 (with color hints), DDColor (tiny / large / artistic),
  DeOldify (artistic / stable), and optionally Stable Diffusion 1.5 + ControlNet.
- Optional restoration before colorizing, chainable: Real-ESRGAN (upscale 2x/4x with
  adjustable denoising), NAFNet (deblur or denoise) and CodeFormer (faces; non-commercial
  license).
- Before/after slider, postprocessing (saturation, temperature, tint, chroma blend, ab
  smoothing) without re-running the model, a model comparison tab, presets, and batch
  processing.
- GPU acceleration through ONNX Runtime (CUDA, DirectML, WebGPU/Vulkan, CoreML), with a
  CPU fallback.

## Download

Prebuilt desktop apps are on the [Releases](https://github.com/Makistos/colorize/releases)
page:

| File | Platform |
|---|---|
| `colorizer-<version>-Linux-cpu.tar.gz` | Linux x86-64, CPU |
| `colorizer-<version>-Windows-directml.zip` | Windows 10/11, any DirectX 12 GPU (falls back to CPU) |

Unpack the archive and run `colorizer` (`colorizer.exe` on Windows). The app opens in its
own window.

The models are not bundled. Each one is downloaded the first time you use it (130–900 MB
per model) from the [`models-v1`](https://github.com/Makistos/colorize/releases/tag/models-v1)
(colorizers) and [`models-v2`](https://github.com/Makistos/colorize/releases/tag/models-v2)
(restorers) releases, checked against a SHA256 hash pinned in the source, and cached in
`~/.cache/colorizer` (override with `COLORIZER_CACHE_DIR`). Expect a pause on the first
run of each model.

The prebuilt apps do not include Stable Diffusion; run from source for that.

## Run from source

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra cpu        # or --extra cuda (NVIDIA) / directml (Windows) / webgpu (AMD etc.)
uv run colorizer-ui        # web UI at http://127.0.0.1:7860
uv run colorizer in/ out/ --model ddcolor --param size=512   # batch CLI
uv run colorizer in/ out/ --restore nafnet,realesrgan --rparam realesrgan.scale=4
```

Add `--extra desktop` and run `uv run colorizer-app` for the native window, or
`--extra diffusion` for Stable Diffusion (about 3.6 GB of extra weights downloaded on first
use).

To build the ONNX models yourself instead of downloading them, see `tools/export_onnx/`.

## License

The application code is released under the MIT license (see `LICENSE`).

The models are by their original authors and keep their own licenses:

| Model | Upstream | License |
|---|---|---|
| Zhang ECCV16, SIGGRAPH17 | [richzhang/colorization](https://github.com/richzhang/colorization) | BSD-2-Clause |
| DDColor | [piddnad/DDColor](https://github.com/piddnad/DDColor) | Apache-2.0 |
| DeOldify | [jantic/DeOldify](https://github.com/jantic/DeOldify) | MIT |
| Real-ESRGAN (restorer) | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause |
| NAFNet (restorer) | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | MIT |
| CodeFormer (restorer) | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | S-Lab License 1.0, **non-commercial use only** |
| YuNet face detector (for CodeFormer) | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) | MIT |
| Stable Diffusion 1.5 + brightness ControlNet | [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [latentcat](https://huggingface.co/latentcat/control_v1p_sd15_brightness) | CreativeML OpenRAIL-M |

The ONNX files in the `models-v1` and `models-v2` releases are conversions of the upstream checkpoints made
with the scripts in `tools/export_onnx/`; the weights themselves are unchanged. The upstream
license texts are in `packaging/licenses/` and attached to those releases.
