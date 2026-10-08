"""Gradio UI.

Widgets are generated from each model's and restorer's ``Param`` schema; this module must
stay free of model-specific code. Inference runs on the background worker; postprocess
changes re-render from the cached ab, and colorizer changes reuse the cached restoration.
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
from typing import Any, TypeVar

# Local-first: no launch telemetry or version pings. Must be set before gradio is imported.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
import numpy as np

from colorizer.core import presets
from colorizer.core.base import ColorizerModel
from colorizer.core.batch import BatchItem, BatchResult, run_batch
from colorizer.core.params import Param
from colorizer.core.pipeline import (
    EXTENSIONS,
    SUFFIX_FORMATS,
    OutputFormat,
    render,
    save_image,
    scale_points,
)
from colorizer.core.postprocess import POSTPROCESS_PARAMS, Postprocess
from colorizer.core.registry import Registry, default_registry
from colorizer.core.restore import (
    RESTORE_PARAMS,
    Restorer,
    RestoreStep,
    validate_restore_params,
)
from colorizer.core.runtime import ENV_DEVICE, available_devices, select_device
from colorizer.core.session import LRU, LoadedImage, Session
from colorizer.core.worker import Cancelled, Job, JobContext, Worker
from colorizer.ui import file_dialog

log = logging.getLogger(__name__)

POLL_SECONDS = 0.1
SETTINGS_FILE = "ui.json"
THUMB_SIZE = 384
T = TypeVar("T")
Rendered = tuple[tuple[np.ndarray, np.ndarray], Path]  # slider pair (uint8 RGB), saved file
# Before/after pairs for the comparison slider.
VIEWS = ("Original | Colorized", "Original | Restored", "Restored | Colorized")


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


def _to_widget(param: Param, value: Any) -> Any:
    """Inverse of ``widget_value``: a validated value as its widget shows it."""
    if param.kind == "points":
        return json.dumps([[x, y, list(rgb)] for x, y, rgb in value])
    return value


def parse_color(value: str) -> tuple[int, int, int]:
    """``#rrggbb`` or ``rgb(a)(r, g, b[, a])`` from gr.ColorPicker -> (r, g, b)."""
    value = value.strip()
    if value.startswith("#") and len(value) in (7, 9):
        return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)
    if value.startswith("rgb"):
        parts = value[value.index("(") + 1 : value.rindex(")")].split(",")
        r, g, b = (round(float(p)) for p in parts[:3])
        return r, g, b
    raise ValueError(f"unrecognised colour {value!r}")


def hex_color(rgb: Sequence[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


@dataclass(frozen=True)
class Slot:
    """One generated param widget of a model (or of a restorer, in ``App.rslots``)."""

    model_id: str
    param: Param


@dataclass(frozen=True)
class Settings:
    params: dict[str, Any]
    post: Postprocess
    restore: list[RestoreStep]


class App:
    def __init__(
        self,
        registry: Registry | None = None,
        session: Session | None = None,
        worker: Worker | None = None,
        output_dir: Path | None = None,
        config_dir: Path | None = None,
        dialogs: Any = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.session = session or Session(self.registry)
        self.worker = worker or Worker()
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="colorizer-ui-"))
        self.models: dict[str, type[ColorizerModel]] = {}
        for model_id in self.registry.ids():
            try:
                cls = self.registry.get_class(model_id)
            except Exception as e:  # broken plugin
                log.warning("model %s unavailable: %s", model_id, e)
                continue
            if cls.available():
                self.models[model_id] = cls
            else:
                log.info("model %s hidden: its optional dependencies are not installed", model_id)
        if not self.models:
            raise RuntimeError("no models available")
        self.slots = [Slot(mid, p) for mid, cls in self.models.items() for p in cls.params]
        self.restorers: dict[str, type[Restorer]] = {}
        self.coming_restorers: list[str] = []  # registered stubs, shown as "not yet"
        for rid in self.registry.restorers.ids():
            try:
                rcls = self.registry.restorers.get_class(rid)
            except Exception as e:  # broken plugin
                log.warning("restorer %s unavailable: %s", rid, e)
                continue
            if not rcls.enabled:
                self.coming_restorers.append(rcls.display_name)
            elif rcls.available():
                self.restorers[rid] = rcls
        # Each restorer's own params, then the shared ones (blend).
        self.rslots = [
            Slot(rid, p)
            for rid, rcls in self.restorers.items()
            for p in (*rcls.params, *RESTORE_PARAMS)
        ]
        self._before: LRU[str, np.ndarray] = LRU(4)
        self._current: Job[Any] | None = None
        # Anything with available() / ask_save_path() / ask_directory() like file_dialog;
        # the desktop app passes the webview's native dialogs.
        self.dialogs = dialogs or file_dialog
        self.config_dir = config_dir or presets.config_dir()
        self.presets_dir = self.config_dir / "presets"
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

    def n_values(self) -> int:
        """Length of the flat value list: model slots, restore ids, restore slots, postprocess."""
        return len(self.slots) + 1 + len(self.rslots) + len(POSTPROCESS_PARAMS)

    def split_all(self, model_id: str, values: Sequence[Any]) -> Settings:
        """Map the flat widget values to validated settings. Raises ``ValueError``."""
        if len(values) != self.n_values():
            raise ValueError(f"expected {self.n_values()} values, got {len(values)}")
        n, m = len(self.slots), len(self.rslots)
        params = {
            s.param.name: widget_value(s.param, v)
            for s, v in zip(self.slots, values[:n], strict=True)
            if s.model_id == model_id
        }
        if model_id not in self.models:
            raise ValueError(f"unknown model {model_id!r}")
        params = self.models[model_id].validate_params(params)
        restore_ids = values[n] or []
        rvalues = values[n + 1 : n + 1 + m]
        restore = []
        for rid in restore_ids:
            if rid not in self.restorers:
                raise ValueError(f"unknown restorer {rid!r}")
            own = {
                s.param.name: widget_value(s.param, v)
                for s, v in zip(self.rslots, rvalues, strict=True)
                if s.model_id == rid
            }
            restore.append(RestoreStep(rid, validate_restore_params(self.restorers[rid], own)))
        post = Postprocess.from_mapping(
            {p.name: v for p, v in zip(POSTPROCESS_PARAMS, values[n + 1 + m :], strict=True)}
        )
        return Settings(params, post, restore)

    def split_values(
        self, model_id: str, values: Sequence[Any]
    ) -> tuple[dict[str, Any], Postprocess]:
        """Model params and postprocess from the flat widget values. Raises ``ValueError``."""
        settings = self.split_all(model_id, values)
        return settings.params, settings.post

    def model_params_for(
        self, model_id: str, params: dict[str, Any], image: LoadedImage, restored: LoadedImage
    ) -> dict[str, Any]:
        """Params for colorizing ``restored``: hint points move with any resolution change."""
        schema = self.models[model_id].params
        return scale_points(schema, params, image.src.size, restored.src.size)

    def _gray(self, image: LoadedImage) -> np.ndarray:
        """The image as neutral gray RGB uint8 (cached)."""
        gray = self._before.get(image.digest)
        if gray is None:
            gray = to_uint8(render(image.src, np.zeros((1, 1, 2), np.float32)))
            self._before.put(image.digest, gray)
        return gray

    def _render(
        self,
        image: LoadedImage,
        restored: LoadedImage,
        ab: np.ndarray,
        post: Postprocess,
        name: str,
        fmt: str,
        quality: int,
        view: str = VIEWS[0],
    ) -> Rendered:
        """Colorize ``restored`` with ``ab``, save it, and pick the slider pair for ``view``."""
        src = restored.src
        rgb = render(src, ab, post)
        out = self.output_dir / f"{Path(name).stem}_colorized{EXTENSIONS[fmt]}"
        save_image(rgb, out, fmt, src.bit_depth, src.exif, int(quality))  # type: ignore[arg-type]
        h, w = src.size
        original = self._gray(image)
        if original.shape[:2] != (h, w):  # restoration changed the size; display only
            original = cv2.resize(original, (w, h), interpolation=cv2.INTER_CUBIC)
        images = {
            "Original": original,
            "Restored": self._gray(restored),
            "Colorized": to_uint8(rgb),
        }
        left, _, right = (view if view in VIEWS else VIEWS[0]).partition(" | ")
        return (images[left], images[right]), out

    def restore_status(self, steps: Sequence[RestoreStep]) -> str:
        names = [self.restorers[s.id].display_name for s in steps]
        return " → ".join(names) + " → " if names else ""

    # --- event handlers -------------------------------------------------------------------

    def colorize(
        self,
        image_path: str | None,
        model_id: str,
        device_name: str,
        fmt: str,
        quality: float,
        view: str,
        *values: Any,
    ) -> Iterator[tuple[Any, Any, str]]:
        """Restore (optional) and colorize on the worker, streaming progress into the status."""
        if not image_path:
            raise gr.Error("Upload an image first.")
        try:
            settings = self.split_all(model_id, values)
            device = select_device(device_name)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        path = Path(image_path)

        def job_fn(ctx: JobContext) -> Rendered:
            ctx.progress(0.0, "Loading image")
            image = self.session.load(path)
            restored = self.session.get_restored(image, settings.restore, device, ctx)
            params = self.model_params_for(model_id, settings.params, image, restored)
            ab = self.session.get_ab(restored, model_id, params, device, ctx)
            ctx.progress(0.9, "Rendering")
            return self._render(
                image, restored, ab, settings.post, path.name, fmt, int(quality), view
            )

        job = self._submit(job_fn)
        try:
            for progress in self._follow(job):
                yield gr.skip(), gr.skip(), progress
            try:
                pair, out = self._result(job)
            except Cancelled:
                yield gr.skip(), gr.skip(), "Cancelled."
                return
            chain = self.restore_status(settings.restore)
            yield pair, str(out), f"Done: {chain}{self.models[model_id].display_name}."
        finally:
            job.cancel()  # no-op when finished; stops the job if Gradio closed this generator

    # --- job plumbing ---------------------------------------------------------------------

    def _submit(self, fn: Any) -> Job[Any]:
        """Run ``fn(ctx)`` on the worker as the current job (replacing any running one)."""
        if self._current is not None:
            self._current.cancel()
        self._current = self.worker.submit(fn)
        return self._current

    @staticmethod
    def _follow(job: Job[Any]) -> Iterator[str]:
        """Yield a status line whenever the job's progress changes, until it finishes."""
        shown: tuple[float, str] | None = None
        while not job.wait(POLL_SECONDS):
            if job.progress != shown:
                shown = job.progress
                yield f"{shown[1]}… {shown[0]:.0%}"

    @staticmethod
    def _result(job: Job[T]) -> T:
        """The job's result; surfaces warnings, maps errors to ``gr.Error``. ``Cancelled``
        propagates."""
        try:
            return job.result()
        except Cancelled:
            raise
        except Exception as e:
            log.exception("job failed")
            raise gr.Error(str(e)) from e
        finally:
            for message in job.warnings:
                gr.Warning(message)

    # --- compare and batch ----------------------------------------------------------------

    def compare(
        self, image_path: str | None, model_ids: list[str], device_name: str, *values: Any
    ) -> Iterator[tuple[Any, str]]:
        """Run the image through several models (each with its current settings)."""
        if not image_path:
            raise gr.Error("Upload an image first.")
        if not model_ids:
            raise gr.Error("Select at least one model to compare.")
        try:
            settings = {mid: self.split_all(mid, values) for mid in model_ids}
            device = select_device(device_name)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        path = Path(image_path)
        steps = next(iter(settings.values())).restore  # shared by all models

        def job_fn(ctx: JobContext) -> list[tuple[np.ndarray, str]]:
            image = self.session.load(path)
            restored = self.session.get_restored(image, steps, device)  # once, then cached
            results = []
            for i, (mid, s) in enumerate(settings.items()):
                name = self.models[mid].display_name
                ctx.progress(i / len(settings), f"{i + 1}/{len(settings)}: {name}")
                params = self.model_params_for(mid, s.params, image, restored)
                ab = self.session.get_ab(restored, mid, params, device)
                rgb = render(restored.src, ab, s.post)
                results.append((to_uint8(rgb), self.caption(mid, s.params)))
            return results

        job = self._submit(job_fn)
        try:
            for progress in self._follow(job):
                yield gr.skip(), progress
            try:
                results = self._result(job)
            except Cancelled:
                yield gr.skip(), "Cancelled."
                return
            yield results, f"Compared {len(results)} models."
        finally:
            job.cancel()

    def caption(self, model_id: str, params: dict[str, Any]) -> str:
        shown = ", ".join(f"{k}={v}" for k, v in params.items() if not isinstance(v, list))
        name = self.models[model_id].display_name
        return f"{name} ({shown})" if shown else name

    def batch(
        self,
        files: list[str] | None,
        model_id: str,
        device_name: str,
        fmt: str,
        quality: float,
        out_dir: str,
        overwrite: bool,
        *values: Any,
    ) -> Iterator[tuple[Any, str]]:
        """Colorize uploaded files into ``out_dir`` as ``<name>_colorized.<ext>``."""
        if not files:
            raise gr.Error("Add some images to the batch first.")
        if not out_dir.strip():
            raise gr.Error("Choose an output folder.")
        try:
            settings = self.split_all(model_id, values)
            device = select_device(device_name)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        folder = Path(out_dir.strip()).expanduser()
        out_fmt: OutputFormat = fmt  # type: ignore[assignment]
        items = [
            BatchItem(Path(f), folder / f"{Path(f).stem}_colorized{EXTENSIONS[fmt]}", out_fmt)
            for f in files
        ]

        def job_fn(ctx: JobContext) -> BatchResult:
            return run_batch(
                items,
                lambda: self.registry.get(model_id, device),
                settings.params,
                settings.post,
                quality=int(quality),
                overwrite=overwrite,
                ctx=ctx,
                restore=settings.restore,
                get_restorer=lambda rid: self.registry.restorers.get(rid, device),
            )

        job = self._submit(job_fn)
        try:
            for progress in self._follow(job):
                yield gr.skip(), progress
            try:
                result = self._result(job)
            except Cancelled:
                yield gr.skip(), "Cancelled (files finished so far were kept)."
                return
            if result.done or result.skipped:
                self._remember_save_dir(folder)
            lines = [f"{result.summary()} → `{folder}`"]
            lines += [f"- skipped (exists): {i.dst.name}" for i in result.skipped]
            lines += [f"- **failed** {i.src.name}: {err}" for i, err in result.failed]
            yield self._thumbnails(result.done), "\n".join(lines)
        finally:
            job.cancel()

    def _thumbnails(self, items: list[BatchItem]) -> list[tuple[str, str]]:
        """Small copies in the served temp dir (outputs may live anywhere on disk)."""
        from PIL import Image

        thumbs = []
        for item in items:
            thumb = self.output_dir / "thumbs" / f"{item.dst.stem}.png"
            thumb.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(item.dst) as im:
                im.thumbnail((THUMB_SIZE, THUMB_SIZE))
                im.convert("RGB").save(thumb)
            thumbs.append((str(thumb), item.dst.name))
        return thumbs

    def suggest_batch_dir(self) -> str:
        return str(self.last_save_dir / "colorized")

    def browse_batch_dir(self, current: str) -> tuple[Any, Any]:
        start = Path(current.strip()).expanduser() if current.strip() else self.last_save_dir
        while not start.is_dir() and start != start.parent:
            start = start.parent
        try:
            chosen = self.dialogs.ask_directory(start)
        except file_dialog.DialogUnavailable as e:
            return gr.skip(), f"{e}. Type the folder path instead."
        return (str(chosen), "") if chosen else (gr.skip(), "")

    # --- presets --------------------------------------------------------------------------

    def preset_names(self) -> list[str]:
        return presets.list_presets(self.presets_dir)

    def save_preset(self, name: str, model_id: str, *values: Any) -> tuple[Any, str]:
        try:
            name = presets.check_name(name)
            s = self.split_all(model_id, values)
        except ValueError as e:
            return gr.skip(), f"Can't save preset: {e}."
        preset = presets.Preset(model_id, s.params, s.post.to_dict(), tuple(s.restore))
        presets.save_preset(name, preset, self.presets_dir)
        return gr.update(choices=self.preset_names(), value=name), f"Saved preset **{name}**."

    def load_preset(self, name: str | None) -> list[Any]:
        """Set the model, its widgets, the restore chain and postprocess from a preset.

        Widgets of other models and of restorers not in the chain are left untouched.
        """
        n_out = 1 + self.n_values()
        if not name:
            return [gr.skip()] * n_out + ["Pick a preset to load."]
        try:
            preset = presets.load_preset(name, self.presets_dir).validate(self.registry)
        except ValueError as e:
            return [gr.skip()] * n_out + [f"Can't load preset: {e}"]
        if preset.model not in self.models:
            return [gr.skip()] * n_out + [f"Preset uses unavailable model {preset.model!r}."]
        missing = [s.id for s in preset.restore if s.id not in self.restorers]
        if missing:
            return [gr.skip()] * n_out + [f"Preset uses unavailable restorer(s) {missing}."]
        chain = {s.id: s.params for s in preset.restore}
        rslot_values = [
            _to_widget(s.param, chain[s.model_id][s.param.name])
            if s.model_id in chain
            else gr.skip()
            for s in self.rslots
        ]
        slot_values = [
            _to_widget(s.param, preset.params[s.param.name])
            if s.model_id == preset.model
            else gr.skip()
            for s in self.slots
        ]
        post_values = [preset.postprocess[p.name] for p in POSTPROCESS_PARAMS]
        status = f"Loaded preset **{name}**. Press **Colorize** to apply."
        restore_ids = [s.id for s in preset.restore]
        return [preset.model, *slot_values, restore_ids, *rslot_values, *post_values, status]

    def delete_preset(self, name: str | None) -> tuple[Any, str]:
        if not name:
            return gr.skip(), "Pick a preset to delete."
        try:
            presets.delete_preset(name, self.presets_dir)
        except ValueError as e:
            return gr.skip(), str(e)
        return gr.update(choices=self.preset_names(), value=None), f"Deleted preset **{name}**."

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
            settings = self.split_all(model_id, values)
            path, out_fmt = self.resolve_save_path(dest, image_path, fmt)
        except ValueError as e:
            return gr.skip(), f"Can't save: {e}."
        if path.exists() and not overwrite:
            return str(path), (
                f"**{path.name}** already exists in {path.parent}. "
                "Tick *Overwrite* or choose another name."
            )
        image = self.session.load(Path(image_path))
        restored = self.session.cached_restored(image, settings.restore)
        params = self.model_params_for(model_id, settings.params, image, restored or image)
        ab = None if restored is None else self.session.cached_ab(restored, model_id, params)
        if restored is None or ab is None:
            return str(path), "Press **Colorize** first: the current settings have no result."
        src = restored.src
        try:
            save_image(
                render(src, ab, settings.post),
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
            chosen = self.dialogs.ask_save_path(initial)
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
        self,
        image_path: str | None,
        model_id: str,
        fmt: str,
        quality: float,
        view: str,
        *values: Any,
    ) -> tuple[Any, Any, Any]:
        """Re-apply postprocess/output/view settings from the caches. Never runs a model."""
        if not image_path:
            return gr.skip(), gr.skip(), gr.skip()
        try:
            settings = self.split_all(model_id, values)
        except ValueError as e:
            return gr.skip(), gr.skip(), f"Invalid setting: {e}"
        path = Path(image_path)
        image = self.session.load(path)
        restored = self.session.cached_restored(image, settings.restore)
        if restored is None:
            return gr.skip(), gr.skip(), "Press **Colorize** to apply the current restore settings."
        params = self.model_params_for(model_id, settings.params, image, restored)
        ab = self.session.cached_ab(restored, model_id, params)
        if ab is None:
            return gr.skip(), gr.skip(), "Press **Colorize** to apply the current model settings."
        pair, out = self._render(
            image, restored, ab, settings.post, path.name, fmt, int(quality), view
        )
        return pair, str(out), "Updated (no re-inference)."

    def select_model(self, model_id: str) -> list[Any]:
        return [
            *(gr.update(visible=mid == model_id) for mid in self.models),
            self.model_info(model_id),
            gr.update(visible=self.has_points(model_id)),
        ]

    # --- point hints (any model with a "points" param) ------------------------------------

    def point_slots(self) -> list[int]:
        """Indices into ``self.slots`` of every ``points`` param."""
        return [i for i, s in enumerate(self.slots) if s.param.kind == "points"]

    def has_points(self, model_id: str) -> bool:
        return any(self.slots[i].model_id == model_id for i in self.point_slots())

    def add_point(self, current: str, x: int, y: int, color: str) -> str:
        """Append ``(x, y, color)`` to a points widget's JSON value."""
        slot = next(self.slots[i] for i in self.point_slots())
        points = widget_value(Param("p", "points", []), current) if current else []
        points.append((int(x), int(y), parse_color(color)))
        return _to_widget(slot.param, points)

    @staticmethod
    def remove_point(current: str, index: int) -> str:
        points = json.loads(current) if current.strip() else []
        if 0 <= index < len(points):
            points.pop(index)
        return json.dumps(points)

    def model_info(self, model_id: str) -> str:
        cls = self.models[model_id]
        return f"**{cls.display_name}** · license: {cls.license}"

    def select_restorers(self, restore_ids: list[str] | None) -> list[Any]:
        chosen = set(restore_ids or [])
        return [gr.update(visible=rid in chosen) for rid in self.restorers]

    def restorer_info(self, rid: str) -> str:
        rcls = self.restorers[rid]
        text = f"**{rcls.display_name}** · license: {rcls.license}"
        if rcls.warning:
            text += f"\n\n⚠️ **{rcls.warning}**"
        return text

    # --- layout ---------------------------------------------------------------------------

    def build(self) -> gr.Blocks:
        first = next(iter(self.models))
        devices = ["auto"] + [d.name for d in available_devices()]
        env_device = os.environ.get(ENV_DEVICE, "auto").lower()
        dialogs = self.dialogs.available()

        with gr.Blocks(title="Colorizer", analytics_enabled=False) as demo:
            with gr.Row():
                # Left: the photo and every setting; shared by all tabs on the right.
                with gr.Column(scale=1, min_width=320):
                    image = gr.Image(
                        type="filepath",
                        label="Black & white photo",
                        height=320,
                        sources=["upload", "clipboard"],
                    )
                    with gr.Accordion("Restore (optional, before colorizing)", open=True):
                        coming = (
                            f" Not implemented yet: {', '.join(self.coming_restorers)}."
                            if self.coming_restorers
                            else ""
                        )
                        restore_dd = gr.Dropdown(
                            choices=[(c.display_name, rid) for rid, c in self.restorers.items()],
                            value=[],
                            multiselect=True,
                            label="Restorers",
                            info="Run in the order selected; results are cached." + coming,
                        )
                        restore_groups = []
                        restore_widgets: list[gr.components.Component] = []
                        for rid, rcls in self.restorers.items():
                            with gr.Group(visible=False) as rgroup:
                                gr.Markdown(self.restorer_info(rid))
                                restore_widgets.extend(
                                    widget_for(p) for p in (*rcls.params, *RESTORE_PARAMS)
                                )
                            restore_groups.append(rgroup)
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
                    point_widgets = [model_widgets[i] for i in self.point_slots()]
                    with gr.Group(visible=self.has_points(first)) as hints_group:
                        gr.Markdown(
                            "**Colour hints:** pick a colour, then click the photo where it "
                            "belongs. The result updates after each change."
                        )
                        with gr.Row(equal_height=True):
                            hint_color = gr.ColorPicker("#3c78c8", label="Hint colour", scale=1)
                            clear_hints_btn = gr.Button("Clear hints", scale=1)

                        @gr.render(inputs=[model_dd, *point_widgets])
                        def hint_list(model_id: str, *values: str) -> None:
                            for slot_index, widget, value in zip(
                                self.point_slots(), point_widgets, values, strict=True
                            ):
                                if self.slots[slot_index].model_id != model_id:
                                    continue
                                points = json.loads(value) if value and value.strip() else []
                                if not points:
                                    gr.Markdown("*No hints yet.*")
                                for i, (x, y, rgb) in enumerate(points):
                                    with gr.Row(equal_height=True):
                                        gr.HTML(
                                            f'<span style="display:inline-block;width:1.2em;'
                                            f"height:1.2em;border-radius:3px;vertical-align:middle;"
                                            f'background:{hex_color(rgb)}"></span>'
                                            f"&nbsp; ({x}, {y}) {hex_color(rgb)}"
                                        )
                                        delete = gr.Button("✕", size="sm", scale=0, min_width=40)
                                    delete.click(
                                        lambda current, i=i: self.remove_point(current, i),
                                        widget,
                                        widget,
                                        api_name=False,
                                    )

                    with gr.Row():
                        run_btn = gr.Button("Colorize", variant="primary")
                        cancel_btn = gr.Button("Cancel", variant="stop")
                    with gr.Accordion("Postprocess", open=True):
                        post_widgets = [widget_for(p) for p in POSTPROCESS_PARAMS]
                    with gr.Accordion("Presets", open=False):
                        preset_dd = gr.Dropdown(
                            self.preset_names(), value=None, label="Preset", interactive=True
                        )
                        with gr.Row():
                            load_preset_btn = gr.Button("Load")
                            delete_preset_btn = gr.Button("Delete")
                        with gr.Row(equal_height=True):
                            preset_name = gr.Textbox(label="Save current settings as", scale=3)
                            save_preset_btn = gr.Button("Save preset", scale=1)
                    with gr.Accordion("Output & device", open=False):
                        device_dd = gr.Dropdown(
                            devices,
                            value=env_device if env_device in devices else "auto",
                            label="Device",
                        )
                        fmt_dd = gr.Dropdown(list(EXTENSIONS), value="png", label="Format")
                        quality = gr.Slider(50, 100, value=95, step=1, label="JPEG quality")
                with gr.Column(scale=2):
                    status = gr.Markdown()
                    with gr.Tabs():
                        with gr.Tab("Result"):
                            # Lossless preview: Gradio's default WebP is lossy, and right-click
                            # "Save image as" saves these bytes. The download button gives the
                            # real output file in the chosen format.
                            view = gr.Radio(
                                list(VIEWS), value=VIEWS[0], label="Compare", show_label=False
                            )
                            slider = gr.ImageSlider(
                                label="Before / after", type="numpy", format="png", max_height=720
                            )
                            with gr.Row():
                                download = gr.DownloadButton("Download result", value=None)
                            with gr.Row(equal_height=True):
                                save_path = gr.Textbox(
                                    label="Save to",
                                    info="Full path; the extension (.png, .jpg, .tif) sets the "
                                    "format.",
                                    scale=5,
                                )
                                with gr.Column(scale=1, min_width=140):
                                    overwrite = gr.Checkbox(label="Overwrite", value=False)
                                    save_btn = gr.Button("Save")
                                    save_as_btn = gr.Button("Save as…", visible=dialogs)
                        with gr.Tab("Compare models"):
                            compare_models = gr.CheckboxGroup(
                                [(cls.display_name, mid) for mid, cls in self.models.items()],
                                value=list(self.models),
                                label="Models (each uses its current settings on the left)",
                            )
                            compare_btn = gr.Button("Compare", variant="primary")
                            compare_gallery = gr.Gallery(
                                label="Results", columns=3, format="png", object_fit="contain"
                            )
                        with gr.Tab("Batch"):
                            batch_files = gr.File(
                                label="Photos", file_count="multiple", file_types=["image"]
                            )
                            with gr.Row(equal_height=True):
                                batch_dir = gr.Textbox(
                                    self.suggest_batch_dir(),
                                    label="Output folder",
                                    info="Files are saved as <name>_colorized.<format>.",
                                    scale=5,
                                )
                                browse_dir_btn = gr.Button(
                                    "Browse…", visible=dialogs, scale=1, min_width=120
                                )
                            batch_overwrite = gr.Checkbox(label="Overwrite existing files")
                            batch_btn = gr.Button(
                                "Colorize all with current settings", variant="primary"
                            )
                            batch_gallery = gr.Gallery(label="Done", columns=4, format="png")

            values = [*model_widgets, restore_dd, *restore_widgets, *post_widgets]
            outputs = [slider, download, status]

            model_dd.change(
                self.select_model, model_dd, [*groups, model_info, hints_group], api_name=False
            )
            restore_dd.change(self.select_restorers, restore_dd, restore_groups, api_name=False)
            run = run_btn.click(
                self.colorize,
                [image, model_dd, device_dd, fmt_dd, quality, view, *values],
                outputs,
                api_name="colorize",
            )
            compare = compare_btn.click(
                self.compare,
                [image, compare_models, device_dd, *values],
                [compare_gallery, status],
                api_name="compare",
            )
            batch = batch_btn.click(
                self.batch,
                [
                    batch_files,
                    model_dd,
                    device_dd,
                    fmt_dd,
                    quality,
                    batch_dir,
                    batch_overwrite,
                    *values,
                ],
                [batch_gallery, status],
                api_name="batch",
            )
            cancel_btn.click(
                self.cancel, None, status, cancels=[run, compare, batch], api_name="cancel"
            )
            browse_dir_btn.click(
                self.browse_batch_dir, batch_dir, [batch_dir, status], api_name=False
            )

            rerender_inputs = [image, model_dd, fmt_dd, quality, view, *values]
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
            for w in (fmt_dd, quality, view):
                event = w.release if isinstance(w, gr.Slider) else w.change
                event(self.rerender, rerender_inputs, outputs, api_name=False)

            run_inputs = [image, model_dd, device_dd, fmt_dd, quality, view, *values]
            for slot_index, widget in zip(self.point_slots(), point_widgets, strict=True):
                slot_model = self.slots[slot_index].model_id

                def on_click(
                    model_id: str,
                    current: str,
                    color: str,
                    evt: gr.SelectData,
                    slot_model: str = slot_model,
                ) -> Any:
                    if model_id != slot_model or evt.index is None:
                        return gr.skip()
                    x, y = evt.index
                    return self.add_point(current, x, y, color)

                image.select(on_click, [model_dd, widget, hint_color], widget, api_name=False)

                def on_hints_changed(
                    *args: Any, slot_model: str = slot_model
                ) -> Iterator[tuple[Any, Any, str]]:
                    if not args[0] or args[1] != slot_model:
                        yield gr.skip(), gr.skip(), gr.skip()
                        return
                    yield from self.colorize(*args)

                widget.change(on_hints_changed, run_inputs, outputs, api_name=False)
            clear_hints_btn.click(
                lambda *_: ["[]"] * len(point_widgets), None, point_widgets, api_name=False
            )
            image.change(self.suggest_save_path, [image, fmt_dd], save_path, api_name=False)
            fmt_dd.change(self.retarget_extension, [save_path, fmt_dd], save_path, api_name=False)
            save_inputs = [image, model_dd, fmt_dd, quality, save_path, overwrite, *values]
            save_btn.click(self.save_to, save_inputs, [save_path, status], api_name="save")
            save_as_btn.click(self.save_as, save_inputs, [save_path, status], api_name=False)

            save_preset_btn.click(
                self.save_preset, [preset_name, model_dd, *values], [preset_dd, status]
            )
            load_preset_btn.click(
                self.load_preset, preset_dd, [model_dd, *values, status], api_name="load_preset"
            )
            delete_preset_btn.click(self.delete_preset, preset_dd, [preset_dd, status])
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
