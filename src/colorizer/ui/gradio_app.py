"""Gradio UI.

Widgets are generated from each model's ``Param`` schema; this module must stay free of
model-specific code. Inference runs on the background worker; postprocess changes
re-render from the cached ab without touching the model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Local-first: no launch telemetry or version pings. Must be set before gradio is imported.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np

from colorizer.core.base import ColorizerModel
from colorizer.core.params import Param
from colorizer.core.pipeline import EXTENSIONS, SUFFIX_FORMATS, render, save_image
from colorizer.core.postprocess import POSTPROCESS_PARAMS, Postprocess
from colorizer.core.registry import Registry, default_registry
from colorizer.core.runtime import ENV_DEVICE, available_devices, select_device
from colorizer.core.session import LRU, LoadedImage, Session
from colorizer.core.worker import Cancelled, Job, JobContext, Worker
from colorizer.ui import file_dialog

log = logging.getLogger(__name__)

POLL_SECONDS = 0.1
SETTINGS_FILE = "ui.json"
Rendered = tuple[np.ndarray, np.ndarray, Path]  # before, after (uint8 RGB), saved file


def widget_for(param: Param) -> gr.components.Component:
    """Build the input widget for a parameter."""
    common: dict[str, Any] = {"label": param.name, "info": param.help or None}
    precision = 0 if param.kind in ("int", "seed") else None
    match param.kind:
        case "int" | "float" if param.min is not None and param.max is not None:
            if param.min == param.max:
                return gr.Number(
                    value=param.default, precision=precision, interactive=False, **common
                )
            step = param.step or (1 if param.kind == "int" else (param.max - param.min) / 100)
            return gr.Slider(
                minimum=param.min, maximum=param.max, step=step, value=param.default, **common
            )
        case "int" | "float" | "seed":
            return gr.Number(
                value=param.default,
                precision=precision,
                minimum=param.min,
                maximum=param.max,
                **common,
            )
        case "bool":
            return gr.Checkbox(value=param.default, **common)
        case "choice":
            assert param.choices is not None
            return gr.Dropdown(choices=list(param.choices), value=param.default, **common)
        case "str":
            return gr.Textbox(value=param.default, **common)
        case "points":
            hint = "JSON: [[x, y, [r, g, b]], ...]"
            common["info"] = f"{param.help} {hint}" if param.help else hint
            return gr.Textbox(value=json.dumps(param.default), **common)
    raise AssertionError(f"unhandled kind {param.kind}")  # pragma: no cover


def widget_value(param: Param, value: Any) -> Any:
    """Convert a widget's value back to the param's type (validation happens later)."""
    if param.kind == "points" and isinstance(value, str):
        return param.parse(value) if value.strip() else []
    if param.kind in ("int", "seed") and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def default_config_dir() -> Path:
    """``$XDG_CONFIG_HOME/colorizer`` or ``~/.config/colorizer``."""
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "colorizer"


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


@dataclass(frozen=True)
class Slot:
    """One generated model-param widget."""

    model_id: str
    param: Param


class App:
    def __init__(
        self,
        registry: Registry | None = None,
        session: Session | None = None,
        worker: Worker | None = None,
        output_dir: Path | None = None,
        config_dir: Path | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.session = session or Session(self.registry)
        self.worker = worker or Worker()
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="colorizer-ui-"))
        self.models: dict[str, type[ColorizerModel]] = {}
        for model_id in self.registry.ids():
            try:
                self.models[model_id] = self.registry.get_class(model_id)
            except Exception as e:  # e.g. an optional extra is not installed
                log.warning("model %s unavailable: %s", model_id, e)
        if not self.models:
            raise RuntimeError("no models available")
        self.slots = [Slot(mid, p) for mid, cls in self.models.items() for p in cls.params]
        self._before: LRU[str, np.ndarray] = LRU(4)
        self._current: Job[Rendered] | None = None
        self.config_dir = config_dir or default_config_dir()
        self.last_save_dir = self._load_last_save_dir()

    # --- settings -------------------------------------------------------------------------

    def _load_last_save_dir(self) -> Path:
        try:
            settings = json.loads((self.config_dir / SETTINGS_FILE).read_text(encoding="utf-8"))
            saved = Path(settings["last_save_dir"])
            if saved.is_dir():
                return saved
        except (OSError, ValueError, KeyError, TypeError):
            pass
        pictures = Path.home() / "Pictures"
        return pictures if pictures.is_dir() else Path.home()

    def _remember_save_dir(self, directory: Path) -> None:
        self.last_save_dir = directory
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            (self.config_dir / SETTINGS_FILE).write_text(
                json.dumps({"last_save_dir": str(directory)}), encoding="utf-8"
            )
        except OSError as e:
            log.warning("could not store settings: %s", e)

    # --- value plumbing -------------------------------------------------------------------

    def split_values(
        self, model_id: str, values: Sequence[Any]
    ) -> tuple[dict[str, Any], Postprocess]:
        """Map the flat widget values (all model slots, then postprocess) to params.

        Raises ``ValueError`` for invalid values.
        """
        n = len(self.slots)
        if len(values) != n + len(POSTPROCESS_PARAMS):
            raise ValueError(f"expected {n + len(POSTPROCESS_PARAMS)} values, got {len(values)}")
        params = {
            s.param.name: widget_value(s.param, v)
            for s, v in zip(self.slots, values[:n], strict=True)
            if s.model_id == model_id
        }
        if model_id not in self.models:
            raise ValueError(f"unknown model {model_id!r}")
        params = self.models[model_id].validate_params(params)
        post = Postprocess.from_mapping(
            {p.name: v for p, v in zip(POSTPROCESS_PARAMS, values[n:], strict=True)}
        )
        return params, post

    def _render(
        self,
        image: LoadedImage,
        ab: np.ndarray,
        post: Postprocess,
        name: str,
        fmt: str,
        quality: int,
    ) -> Rendered:
        src = image.src
        before = self._before.get(image.digest)
        if before is None:
            before = to_uint8(render(src, np.zeros((1, 1, 2), np.float32)))
            self._before.put(image.digest, before)
        rgb = render(src, ab, post)
        out = self.output_dir / f"{Path(name).stem}_colorized{EXTENSIONS[fmt]}"
        save_image(rgb, out, fmt, src.bit_depth, src.exif, int(quality))  # type: ignore[arg-type]
        return before, to_uint8(rgb), out

    # --- event handlers -------------------------------------------------------------------

    def colorize(
        self,
        image_path: str | None,
        model_id: str,
        device_name: str,
        fmt: str,
        quality: float,
        *values: Any,
    ) -> Iterator[tuple[Any, Any, str]]:
        """Run inference on the worker, streaming progress into the status line."""
        if not image_path:
            raise gr.Error("Upload an image first.")
        try:
            params, post = self.split_values(model_id, values)
            device = select_device(device_name)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        path = Path(image_path)

        def job_fn(ctx: JobContext) -> Rendered:
            ctx.progress(0.0, "Loading image")
            image = self.session.load(path)
            ab = self.session.get_ab(image, model_id, params, device, ctx)
            ctx.progress(0.9, "Rendering")
            return self._render(image, ab, post, path.name, fmt, int(quality))

        if self._current is not None:
            self._current.cancel()
        job = self._current = self.worker.submit(job_fn)
        try:
            shown: tuple[float, str] | None = None
            while not job.wait(POLL_SECONDS):
                if job.progress != shown:
                    shown = job.progress
                    yield gr.skip(), gr.skip(), f"{shown[1]}… {shown[0]:.0%}"
            try:
                before, after, out = job.result()
            except Cancelled:
                yield gr.skip(), gr.skip(), "Cancelled."
                return
            except Exception as e:
                log.exception("colorize failed")
                raise gr.Error(str(e)) from e
            finally:
                for message in job.warnings:
                    gr.Warning(message)
            yield (before, after), str(out), f"Done: {self.models[model_id].display_name}."
        finally:
            job.cancel()  # no-op when finished; stops the job if Gradio closed this generator

    def suggest_save_path(self, image_path: str | None, fmt: str) -> Any:
        """Default "Save to" path for a newly loaded image."""
        if not image_path:
            return gr.skip()
        return str(self.last_save_dir / f"{Path(image_path).stem}_colorized{EXTENSIONS[fmt]}")

    def retarget_extension(self, dest: str, fmt: str) -> Any:
        """Keep the "Save to" extension in step with the Format dropdown."""
        path = Path(dest.strip())
        if dest.strip() and path.suffix.lower() in SUFFIX_FORMATS:
            return str(path.with_suffix(EXTENSIONS[fmt]))
        return gr.skip()

    def resolve_save_path(self, dest: str, image_path: str, fmt: str) -> tuple[Path, str]:
        """Turn the "Save to" text into (absolute path, format).

        A directory gets the default file name; the extension decides the format, and a
        missing or unknown extension gets the one for ``fmt``. Raises ``ValueError``.
        """
        text = dest.strip()
        if not text:
            raise ValueError("enter a file name to save to")
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = self.last_save_dir / path
        if path.is_dir() or text.endswith(("/", os.sep)):
            path = path / f"{Path(image_path).stem}_colorized{EXTENSIONS[fmt]}"
        out_fmt = SUFFIX_FORMATS.get(path.suffix.lower())
        if out_fmt is None:
            path = path.with_name(path.name + EXTENSIONS[fmt])
            out_fmt = SUFFIX_FORMATS[EXTENSIONS[fmt]]
        return path, out_fmt

    def save_to(
        self,
        image_path: str | None,
        model_id: str,
        fmt: str,
        quality: float,
        dest: str,
        overwrite: bool,
        *values: Any,
    ) -> tuple[Any, str]:
        """Write the current result to ``dest`` at full quality, from the cached ab."""
        if not image_path:
            return gr.skip(), "Upload and colorize an image first."
        try:
            params, post = self.split_values(model_id, values)
            path, out_fmt = self.resolve_save_path(dest, image_path, fmt)
        except ValueError as e:
            return gr.skip(), f"Can't save: {e}."
        if path.exists() and not overwrite:
            return str(path), (
                f"**{path.name}** already exists in {path.parent}. "
                "Tick *Overwrite* or choose another name."
            )
        image = self.session.load(Path(image_path))
        ab = self.session.cached_ab(image, model_id, params)
        if ab is None:
            return str(path), "Press **Colorize** first: the current model settings have no result."
        src = image.src
        try:
            save_image(
                render(src, ab, post),
                path,
                out_fmt,  # type: ignore[arg-type]
                src.bit_depth,
                src.exif,
                int(quality),
            )
        except OSError as e:
            return str(path), f"Can't save: {e}"
        self._remember_save_dir(path.parent)
        return str(path), f"Saved **{path}**"

    def save_as(
        self,
        image_path: str | None,
        model_id: str,
        fmt: str,
        quality: float,
        dest: str,
        overwrite: bool,
        *values: Any,
    ) -> tuple[Any, str]:
        """Ask for a path with the desktop's save dialog, then save there."""
        if not image_path:
            return gr.skip(), "Upload and colorize an image first."
        try:
            initial, _ = self.resolve_save_path(dest, image_path, fmt)
        except ValueError:
            initial = Path(self.suggest_save_path(image_path, fmt))
        try:
            chosen = file_dialog.ask_save_path(initial)
        except file_dialog.DialogUnavailable as e:
            return gr.skip(), f"{e}. Type a path in *Save to* and press **Save**."
        if chosen is None:
            return gr.skip(), "Save cancelled."
        # The dialog already asked about overwriting.
        return self.save_to(image_path, model_id, fmt, quality, str(chosen), True, *values)

    def cancel(self) -> str:
        if self._current is not None:
            self._current.cancel()
        return "Cancelled."

    def rerender(
        self, image_path: str | None, model_id: str, fmt: str, quality: float, *values: Any
    ) -> tuple[Any, Any, Any]:
        """Re-apply postprocess/output settings using the cached ab. Never runs the model."""
        if not image_path:
            return gr.skip(), gr.skip(), gr.skip()
        try:
            params, post = self.split_values(model_id, values)
        except ValueError as e:
            return gr.skip(), gr.skip(), f"Invalid setting: {e}"
        path = Path(image_path)
        image = self.session.load(path)
        ab = self.session.cached_ab(image, model_id, params)
        if ab is None:
            return gr.skip(), gr.skip(), "Press **Colorize** to apply the current model settings."
        before, after, out = self._render(image, ab, post, path.name, fmt, int(quality))
        return (before, after), str(out), "Updated (no re-inference)."

    def select_model(self, model_id: str) -> list[Any]:
        return [gr.update(visible=mid == model_id) for mid in self.models] + [
            self.model_info(model_id)
        ]

    def model_info(self, model_id: str) -> str:
        cls = self.models[model_id]
        return f"**{cls.display_name}** · license: {cls.license}"

    # --- layout ---------------------------------------------------------------------------

    def build(self) -> gr.Blocks:
        first = next(iter(self.models))
        devices = ["auto"] + [d.name for d in available_devices()]
        env_device = os.environ.get(ENV_DEVICE, "auto").lower()

        with gr.Blocks(title="Colorizer", analytics_enabled=False) as demo:
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    image = gr.Image(
                        type="filepath",
                        label="Black & white photo",
                        height=320,
                        sources=["upload", "clipboard"],
                    )
                    model_dd = gr.Dropdown(
                        choices=[(cls.display_name, mid) for mid, cls in self.models.items()],
                        value=first,
                        label="Model",
                    )
                    model_info = gr.Markdown(self.model_info(first))
                    groups = []
                    model_widgets: list[gr.components.Component] = []
                    for mid, cls in self.models.items():
                        with gr.Group(visible=mid == first) as group:
                            model_widgets.extend(widget_for(p) for p in cls.params)
                        groups.append(group)
                    with gr.Row():
                        run_btn = gr.Button("Colorize", variant="primary")
                        cancel_btn = gr.Button("Cancel", variant="stop")
                    with gr.Accordion("Postprocess", open=True):
                        post_widgets = [widget_for(p) for p in POSTPROCESS_PARAMS]
                    with gr.Accordion("Output & device", open=False):
                        device_dd = gr.Dropdown(
                            devices,
                            value=env_device if env_device in devices else "auto",
                            label="Device",
                        )
                        fmt_dd = gr.Dropdown(list(EXTENSIONS), value="png", label="Format")
                        quality = gr.Slider(50, 100, value=95, step=1, label="JPEG quality")
                with gr.Column(scale=2):
                    # Lossless preview: Gradio's default WebP is lossy, and right-click
                    # "Save image as" saves these bytes. The download button gives the
                    # real output file in the chosen format.
                    slider = gr.ImageSlider(
                        label="Before / after", type="numpy", format="png", max_height=720
                    )
                    with gr.Row():
                        download = gr.DownloadButton("Download result", value=None)
                    with gr.Row(equal_height=True):
                        save_path = gr.Textbox(
                            label="Save to",
                            info="Full path; the extension (.png, .jpg, .tif) sets the format.",
                            scale=5,
                        )
                        with gr.Column(scale=1, min_width=140):
                            overwrite = gr.Checkbox(label="Overwrite", value=False)
                            save_btn = gr.Button("Save")
                            save_as_btn = gr.Button("Save as…", visible=file_dialog.available())
                    status = gr.Markdown()

            values = [*model_widgets, *post_widgets]
            outputs = [slider, download, status]

            model_dd.change(self.select_model, model_dd, [*groups, model_info], api_name=False)
            run = run_btn.click(
                self.colorize,
                [image, model_dd, device_dd, fmt_dd, quality, *values],
                outputs,
                api_name="colorize",
            )
            cancel_btn.click(self.cancel, None, status, cancels=[run], api_name="cancel")
            rerender_inputs = [image, model_dd, fmt_dd, quality, *values]
            # Sliders fire on release, not on every drag step. The first one also exposes
            # the named "rerender" API endpoint (used by scripts and tests).
            for i, w in enumerate(post_widgets):
                event = w.release if isinstance(w, gr.Slider) else w.change
                event(
                    self.rerender,
                    rerender_inputs,
                    outputs,
                    api_name="rerender" if i == 0 else False,
                )
            image.change(self.suggest_save_path, [image, fmt_dd], save_path, api_name=False)
            fmt_dd.change(self.retarget_extension, [save_path, fmt_dd], save_path, api_name=False)
            save_inputs = [image, model_dd, fmt_dd, quality, save_path, overwrite, *values]
            save_btn.click(self.save_to, save_inputs, [save_path, status], api_name="save")
            save_as_btn.click(self.save_as, save_inputs, [save_path, status], api_name=False)
            for w in (fmt_dd, quality):
                event = w.release if isinstance(w, gr.Slider) else w.change
                event(self.rerender, rerender_inputs, outputs, api_name=False)
        return demo


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="colorizer-ui", description="Colorizer web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("colorizer").setLevel(logging.INFO)
    app = App()
    app.build().queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(app.output_dir)],
    )


if __name__ == "__main__":
    main()
