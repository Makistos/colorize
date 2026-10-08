# Colorizer — Project Spec for Claude Code

AI colorization of black-and-white photographs, with optional restoration (denoise, deblur, upscale, faces) before colorizing. Local-first, cross-platform (Linux primary; Windows/macOS should work). Multiple selectable models, each with editable parameters.

## Stack

- Python 3.11+, managed with `uv`
- UI: Gradio (local web UI); optional desktop wrapper via `pywebview` later
- Inference: ONNX Runtime by default; PyTorch only where unavoidable (diffusion)
- Image I/O: Pillow, OpenCV (`opencv-python-headless`), `scikit-image` for Lab conversion
- Model downloads: `huggingface_hub` (or direct URLs with SHA256 check)
- Tooling: `ruff` (lint + format), `mypy` (strict on `core/`), `pytest`

## Commands

```bash
uv sync --extra cpu           # install; or --extra cuda (NVIDIA) / directml (Windows) / webgpu (AMD etc.); exactly one
uv run colorizer-ui           # launch Gradio UI (http://127.0.0.1:7860)
uv run colorizer in/ out/ --model ddcolor --param size=512   # batch CLI
uv run pytest                 # tests
COLORIZER_TEST_WEIGHTS=1 uv run pytest -m weights   # real-model tests
uv run --group export python tools/export_onnx/zhang.py   # ECCV16 + SIGGRAPH17 ONNX export (CPU torch)
uv run --group export python tools/export_onnx/deoldify.py   # DeOldify, rebuilt without fastai
uv run --group export python tools/export_onnx/ddcolor.py    # DDColor, uses upstream code at a pinned commit
uv run --group export python tools/export_onnx/realesrgan.py # restorers: re-implemented archs, no upstream code run
uv run --group export python tools/export_onnx/nafnet.py
uv run --group export python tools/export_onnx/codeformer.py
uv run colorizer in/ out/ --restore nafnet,realesrgan --rparam realesrgan.scale=4 --rparam nafnet.blend=0.7
uv run ruff check . && uv run ruff format . && uv run mypy src/colorizer/core
```

## Layout

```
src/colorizer/
  core/
    params.py      # Param dataclass + validation
    base.py        # Plugin base; ColorizerModel ABC
    restore.py     # Restorer ABC, shared restore params (blend), RestoreStep
    registry.py    # model + restorer discovery (entry points + built-ins), LRU caches
    pipeline.py    # load → [restore…] → Lab split → infer → recombine → postprocess
    postprocess.py # saturation, temperature, chroma blend, ab denoise
    runtime.py     # device / ONNX execution provider selection
    weights.py     # download, cache (~/.cache/colorizer), checksum
    worker.py      # background job queue, cancellation, progress
    batch.py       # many files, one model/settings (CLI + UI batch)
    presets.py     # named {model, params, postprocess, restore} JSON presets
    session.py     # UI caches: decoded images, restore chain results, ab per image/model/params
  models/
    zhang.py       # ECCV16 + SIGGRAPH17 (with user hints)
    ddcolor.py
    deoldify.py
    sd_controlnet.py   # optional extra: `uv sync --extra diffusion`
    realesrgan.py  # restorers…
    nafnet.py
    codeformer.py
    stubs.py       # swinir, seedvr2: registered, enabled = False
    _onnx.py, _tiles.py  # shared ONNX download/session plumbing, tiled inference
  ui/
    gradio_app.py  # builds widgets from Param schema
  cli.py
tests/
```

## Core contracts

```python
@dataclass(frozen=True)
class Param:
    name: str
    kind: Literal["int", "float", "bool", "choice", "str", "seed", "points"]
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    help: str = ""


class Plugin(ABC):  # shared by models and restorers
    id: ClassVar[str]  # "ddcolor"
    display_name: ClassVar[str]
    params: ClassVar[tuple[Param, ...]]
    license: ClassVar[str]  # shown in UI
    step_callback: Callable[[float], None] | None  # set by the caller; may raise Cancelled

    @abstractmethod
    def load(self, device: Device) -> None: ...
    def unload(self) -> None: ...


class ColorizerModel(Plugin):
    @abstractmethod
    def predict_ab(self, L: np.ndarray, **params) -> np.ndarray:
        """L: float32 HxW in [0,100]. Returns ab: float32 h x w x 2 (any size; pipeline resizes)."""


class Restorer(Plugin):  # core/restore.py
    enabled: ClassVar[bool] = True  # False = registered stub, rejected by validation
    warning: ClassVar[str] = ""  # shown in the UI / logged by the CLI (e.g. non-commercial)

    @abstractmethod
    def restore(self, gray: np.ndarray, **params) -> np.ndarray:
        """gray: float32 HxW in [0, 1] (sRGB-encoded). Returns float32 H'xW' in [0, 1]
        (resolution may change)."""
```

Rules:
- The UI must never contain model- or restorer-specific code. Widgets are generated from `params`.
- Unknown or out-of-range params raise `ValueError` before inference (for a restore chain: before any step runs).
- Models and restorers load lazily on first use and are cached (`Registry`, `Registry.restorers`); LRU-unload when switching if VRAM is tight.
- Restorers are discovered from `BUILTIN_RESTORERS` and the `colorizer.restorers` entry-point group.

## Pipeline (pipeline.py)

1. Load image and apply EXIF orientation. Treat as grayscale even if the file is RGB (old scans are often sepia: convert to L).
2. Optional restore chain (`restore_image`): L → sRGB gray, each restorer in order (each may change the size), then the shared `blend` per step: mix with the bicubic-upscaled input (0–1, default 1) to limit invented detail; gray → L. The restored image replaces the original from here on (its size is the output size). `points` params are rescaled to the restored size (`scale_points`).
3. RGB → Lab. Keep the **full-resolution L**.
4. The model receives the L channel resized to its working size and returns ab.
5. Upscale ab to the (restored) image size (bicubic), recombine with that full-resolution L, and convert Lab → RGB. Out-of-gamut pixels lose chroma at constant hue (`lab_to_rgb`), never lightness: the output L equals the input L.
6. Apply postprocess params (shared across all models):
   - `saturation` float 0–2 (default 1.0): scales ab
   - `temperature` float −1–1, `tint` float −1–1: shift b / a
   - `chroma_blend` float 0–1 (default 1.0): mix ab with zero (grayscale)
   - `ab_smooth` float 0–5: Gaussian blur sigma on ab only
7. Save the result, preserving EXIF. Output formats: PNG, JPEG (quality param), TIFF (16-bit if the input was 16-bit).

Caching (`session.py`): each restore step's result is cached under a key chained from the image hash and every step so far (id + validated params). Colorizer and postprocess changes reuse it; the restored image's key is its identity in the ab cache, so restore changes invalidate ab too.

## Models

Verify weight sources, checkpoints, and licenses before hardcoding them, and record the URL plus SHA256 in each model module. Prefer ONNX: export once with a script in `tools/export_onnx/`, and avoid heavy runtime deps such as fastai.

| id | Source | Params |
|---|---|---|
| `zhang_eccv16` | richzhang/colorization (BSD-2) | `size` int 256 (fixed) |
| `zhang_siggraph17` | same | `size`; `hints` points: list of (x, y, rgb) fed as `input_B` + `mask_B` |
| `ddcolor` | piddnad/DDColor | `variant` choice [tiny, large, artistic]; `size` int 256–1024, step 64, default 512 |
| `deoldify` | jantic/DeOldify (MIT) | `variant` choice [artistic, stable]; `render_factor` int 7–45, default 35 (working size = render_factor × 16) |
| `sd_controlnet` | SD 1.5 + ControlNet recolor/lineart (optional extra) | `prompt`, `negative_prompt`, `steps` 10–50, `cfg` 1–15, `strength` 0–1, `seed` |

ONNX builds are hosted as assets of the `models-v1` (colorizers) and `models-v2` (restorers) GitHub releases (SHA256 in `models/_onnx.py`, `ONNX_SHA256`; release per file in `ONNX_RELEASE`) and downloaded on first use via `core/weights.py`; a local export in the cache dir (`$COLORIZER_CACHE_DIR`, default `~/.cache/colorizer`) takes precedence. `tools/export_onnx/<model>.py` downloads the SHA256-pinned upstream checkpoint and writes the ONNX into the cache dir. Re-exported ONNX files need a new `models-vN` release and updated hashes; upload the upstream license texts in `packaging/licenses/` with them.

Implement them in this order: Zhang ECCV16 → DDColor → DeOldify → SIGGRAPH17 hints → SD.

### Restorers

All take the shared `blend` param too. Gray input is replicated to RGB for these RGB-trained networks and the output averaged back. Upstream SHA256s and URLs are in each module; license texts in `packaging/licenses/`.

| id | Source (license) | Params | Notes |
|---|---|---|---|
| `realesrgan` | xinntao/Real-ESRGAN `realesr-general-x4v3` (BSD-3) | `scale` int 2\|4 (default 2); `tile` int 0–2048 step 32 (0 = off, default 512); `denoise_strength` float 0–1 (default 0.5) | Denoise = upstream DNI between the general and `wdn` weights, done inside the ONNX graph (`denoise` input): ORT ignores initializer overrides for prepacked weights at the default optimization level. Scale 2 = x4 then Lanczos down (upstream `--outscale`). |
| `nafnet` | megvii-research/NAFNet (MIT) | `variant` choice [deblur, denoise] (default denoise) | GoPro-width64 / SIDD-width64; weights from a HF mirror pinned by commit, byte-identical to the official Google Drive files. 384 px tiles (≈ upstream NAFNetLocal), padded to multiples of 16. Denoise slightly smooths already-clean images; use `blend`. |
| `codeformer` | sczhou/CodeFormer (S-Lab License 1.0, **non-commercial**) | `fidelity` float 0–1 (default 0.7); `upscale_bg` bool (default false: same size; true: 2x via Real-ESRGAN) | Faces via OpenCV YuNet (MIT, HF `opencv/face_detection_yunet` pinned), FFHQ 5-point alignment, feathered square paste-back (no face-parsing net), tone-matched like upstream's gray path. `warning` shown in the UI. |
| `swinir`, `seedvr2` | JingyunLiang/SwinIR, ByteDance-Seed/SeedVR (Apache-2.0) | — | Stubs: registered, `enabled = False`. |

Restorer ONNX files are hosted in the `models-v2` release (`ONNX_RELEASE` in `models/_onnx.py` maps files to release tags; default `models-v1`). The export scripts re-implement each architecture (no upstream code is fetched or executed), load the official weights with `strict=True`, and check ONNX against PyTorch (NAFNet also checks that a degraded photo improves; on uniform noise it is ill-conditioned even in float64, so its check uses a real photo).

## Runtime (runtime.py)

- ONNX provider priority: CUDA → ROCm → DirectML → CoreML → WebGPU → CPU. Pick the first one listed by `onnxruntime.get_available_providers()`. WebGPU (extra `webgpu`, Vulkan via Dawn) is the supported path for AMD GPUs on Linux: the ROCm EP was removed upstream and MIGraphX needs a system ROCm install.
- Users can force a device via the `COLORIZER_DEVICE` env var or a UI dropdown.
- If a GPU provider fails on load, fall back to CPU with a visible warning.
- ONNX Runtime comes from exactly one of the conflicting extras `cpu`, `cuda`, `directml`, `webgpu` (they all install the `onnxruntime` module). The `cuda` extra pulls CUDA/cuDNN from PyPI; `runtime.py` calls `onnxruntime.preload_dlls()` so they are found.

## UI (gradio_app.py)

- Inputs: single image, or multi-file upload for batch.
- Model dropdown. Changing it rebuilds the parameter panel (use `gr.render` or visibility-toggled groups).
- A "Postprocess" accordion with the shared params. Postprocessing re-runs **without re-inference** (cache the last ab per image+model+params hash).
- Output: before/after comparison slider (`gr.ImageSlider` if available, else side by side), download button.
- Save to a chosen path: "Save to" textbox (extension sets the format; default `<last dir>/<stem>_colorized.<ext>`), "Save" (refuses to overwrite unless ticked) and "Save as…" (native dialog via kdialog/zenity/tkinter in a subprocess, `ui/file_dialog.py`). Last folder is stored in `~/.config/colorizer/ui.json`.
- "Compare models" tab runs the same image through N selected models and shows a grid.
- Presets: save/load JSON `{model, params, postprocess, restore}` in `~/.config/colorizer/presets/`; `restore` is an optional list of `{id, params}` (older presets load unchanged).
- Restore section above the model dropdown: a multi-select of enabled restorers (chain order = selection order) with a visibility-toggled param group per restorer (its params + `blend`, license, warning); stubs are listed as not implemented. The compare view switch shows Original | Colorized, Original | Restored or Restored | Colorized and re-renders from the caches. Compare runs the chain once for all models; batch runs it per file.
- SIGGRAPH17 hints: click on the image to add a point, pick a color, and see the list of points with delete buttons. Implemented generically for any `points` param; the result re-colorizes after each change. The released weights take the hint mask uncentered (0/1); see `tools/export_onnx/zhang.py`.
- Progress and cancel for long jobs (via worker.py).
- Model param panels are visibility-toggled groups built at startup (fixed API endpoints `/colorize`, `/rerender`, `/cancel`). `/colorize` takes `(image, model, device, format, quality, view, *values)` and `/rerender` `(image, model, format, quality, view, *values)`, where `values` = model slots, restore ids, restore slots, postprocess.
- Gradio telemetry and update checks are disabled (local-first).

## CLI (cli.py)

`colorizer INPUT OUTPUT [--model ID] [--param k=v ...] [--preset FILE] [--device cpu|cuda|...] [--format png|jpg|tiff] [--restore ID[,ID...]] [--rparam ID.KEY=VALUE ...]`

- `--restore` sets the chain (in order; overrides the preset's). `--rparam` sets a restorer param or the shared `blend` (`nafnet.blend=0.5`); the id must be in the chain, and a repeated id shares its params. Restorer warnings are logged. `--list-models` lists restorers.

- INPUT can be a file or a directory (recursive with `-r`).
- Skip existing outputs unless `--overwrite` is given. Exit non-zero if any file failed, and print a summary.

## Testing

- Unit tests: Param validation, Lab round-trip (L preserved bit-exact within tolerance), postprocess math, registry discovery.
- A `DummyModel` (returns constant ab) and a `DummyRestorer` (nearest upscale + offset) for pipeline and UI tests. No network or weights are needed in default tests.
- Integration tests marked `@pytest.mark.weights`, skipped unless `COLORIZER_TEST_WEIGHTS=1`.
- Golden test: a tiny fixture image through each real model checks output shape, dtype, and nonzero chroma; each restorer checks shape/range and that the output still resembles the input; CodeFormer must sharpen the face in a real portrait and leave the rest untouched.

### Test plan

Test layers, from fastest to slowest. Each layer has its own pytest marker so CI can select it.

| Layer | Marker | Needs | Runs in |
|---|---|---|---|
| Unit | (none) | nothing | every push/PR, all OSes |
| Pipeline / CLI (DummyModel) | (none) | nothing | every push/PR, all OSes |
| UI smoke | `ui` | gradio | every push/PR, Linux |
| Model integration | `weights` | downloaded weights (cached) | nightly + manual, Linux CPU |
| Diffusion | `diffusion` | `--extra diffusion` + SD weights | manual only / local GPU |
| GPU | `gpu` | CUDA/ROCm device | local or self-hosted runner only |

**Unit (`tests/unit/`)**
- `params.py`: each `kind` accepts valid values and rejects wrong type, out-of-range, unknown names, and bad choices with `ValueError`. Also check that `int` + `step` snapping and `points` parsing work.
- Lab round-trip: random and fixture images. L channel max abs error < 1e-3 (float) or ≤ 1 LSB (8/16-bit).
- `postprocess.py`: `saturation=0` and `chroma_blend=0` give a gray image. `saturation=1`, `temperature=0`, `tint=0`, and `ab_smooth=0` leave the image unchanged. Temperature and tint move b and a in the correct sign. `ab_smooth` never modifies L.
- `registry.py`: built-ins are discovered. A fake entry point is discovered. Duplicate ids raise an error. The LRU cache unloads the oldest model.
- `runtime.py`: monkeypatch `get_available_providers()` to verify priority order, the `COLORIZER_DEVICE` override, and CPU fallback (with a logged warning) when GPU load raises.
- `weights.py`: serve a local file via `file://` or a tmp HTTP server to check that a SHA256 mismatch raises an error, the cache hit skips the download, and a partial download is not treated as cached.
- `worker.py`: jobs report progress, cancellation stops before the next step, and exceptions propagate to the caller.

**Pipeline / CLI (`tests/pipeline/`, DummyModel)**
- EXIF orientation applied. RGB/sepia input treated as grayscale. Output size equals input size. Full-resolution L preserved.
- Formats: PNG, JPEG (quality honored), TIFF 8- and 16-bit in → 16-bit out. EXIF copied.
- CLI: file and dir input, `-r`, `--overwrite` vs skip, `--param` parsing errors, `--preset` loading, and non-zero exit + summary when one file is corrupt. Use `subprocess`/`CliRunner` on `tmp_path`.
- Postprocess cache: changing postprocess params doesn't call `predict_ab` again (count calls on DummyModel). Changing model params does.

**UI smoke (`tests/ui/`, marker `ui`)**
- Build the Blocks app without launching. Every registered model's params map to a widget of the right type (contract: no model-specific UI code).
- Use `gradio_client` against `app.launch(prevent_thread_lock=True)` on a random port with DummyModel. Check that colorizing returns an image, changing postprocess does not re-infer, and preset save/load round-trips in a tmp `XDG_CONFIG_HOME`.
- Manual checklist per release (not automated): slider, hint clicking, compare grid, cancel button.

**Model integration (marker `weights`, `COLORIZER_TEST_WEIGHTS=1`)**
- Golden test per model on a 64×64 and a 300×200 (non-square) fixture: shape, dtype, nonzero chroma, and no NaN.
- Regression: compare against a stored reference output with a tolerance (mean ΔE < 2) to catch export or preprocessing drift. Regenerate with `pytest --update-golden`.
- ONNX export scripts in `tools/export_onnx/`: the exported model matches the PyTorch output within tolerance (run only when the scripts change).

**Quality gates**: `ruff check`, `ruff format --check`, and `mypy --strict src/colorizer/core` must pass. Coverage target ≥ 85% on `core/` (`pytest-cov`), measured on the default (no-weights) suite.

### CI (GitHub Actions)

- `ci.yml` (push + PR): lint/type job on Linux, plus a test matrix of {ubuntu, windows, macos} × Python {3.11, 3.13} running default + `ui` markers. Use `astral-sh/setup-uv` with caching enabled.
- `weights.yml` (nightly cron + `workflow_dispatch`): Linux CPU, `COLORIZER_TEST_WEIGHTS=1`. Cache `~/.cache/colorizer` with `actions/cache` keyed on the hash of the SHA256 list in `models/`. Run all models except diffusion.
- GPU and diffusion tests are not run on GitHub-hosted runners (no free GPUs). Run them locally before a release, or on a self-hosted runner.
- `build.yml` (tag `v*` + `workflow_dispatch`): PyInstaller builds (Linux CPU, Windows DirectML); on a tag, archives are published as a GitHub Release.

## Conventions

- Type hints everywhere; dataclasses over dicts for structured data.
- No global mutable state except the model cache in the registry.
- Logging via `logging`, not print. The UI surfaces warnings to the user.
- Keep `core/` free of Gradio imports.
- Small, focused commits per milestone; update this file when contracts change.

## Milestones (acceptance criteria)

Status: milestones 1-5 done (2026-10-07); restore stage added (2026-10-08). Cancellation is cooperative between pipeline steps; a running ONNX inference call is not interrupted.

1. **Core + Zhang ECCV16 on CPU**: the CLI colorizes a JPEG end to end and tests pass.
2. **Gradio UI**: model params are auto-generated, the before/after view works, and postprocess changes don't re-run inference.
3. **DDColor + DeOldify via ONNX**: GPU is auto-detected and CPU fallback works.
4. **Batch, presets, compare tab, SIGGRAPH17 hints.**
5. **Optional SD ControlNet extra and pywebview packaging** (PyInstaller build for Linux; Windows if feasible).
