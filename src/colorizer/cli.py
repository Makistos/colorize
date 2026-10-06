"""Batch command-line interface.

colorizer INPUT OUTPUT [--model ID] [--param k=v ...] [--preset FILE] [--device NAME]
          [--format png|jpg|tiff] [-r] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colorizer.core import pipeline
from colorizer.core.params import parse_params
from colorizer.core.pipeline import EXTENSIONS, INPUT_SUFFIXES, OutputFormat
from colorizer.core.postprocess import POSTPROCESS_PARAMS, Postprocess
from colorizer.core.registry import Registry, default_registry
from colorizer.core.runtime import select_device

log = logging.getLogger("colorizer")

DEFAULT_MODEL = "zhang_eccv16"
_SUFFIX_FORMATS: dict[str, OutputFormat] = {
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpg",
    ".tif": "tiff",
    ".tiff": "tiff",
}


@dataclass
class Settings:
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    post: Postprocess = field(default_factory=Postprocess)


@dataclass
class Summary:
    done: int = 0
    skipped: int = 0
    failed: list[Path] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colorizer", description="Colorize black-and-white photos.")
    p.add_argument("input", type=Path, nargs="?", help="image file or directory")
    p.add_argument("output", type=Path, nargs="?", help="output file or directory")
    p.add_argument("--model", help=f"model id (default: {DEFAULT_MODEL} or the preset's)")
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="K=V",
        help="model or postprocess parameter; repeatable",
    )
    p.add_argument("--preset", type=Path, help="JSON preset {model, params, postprocess}")
    p.add_argument("--device", help="auto, cpu, cuda, rocm, directml, coreml")
    p.add_argument("--format", choices=list(EXTENSIONS), help="output format")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    p.add_argument("-r", "--recursive", action="store_true", help="recurse into directories")
    p.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    p.add_argument("--list-models", action="store_true", help="list models and their params")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def resolve_settings(args: argparse.Namespace, registry: Registry) -> Settings:
    """Merge preset and CLI arguments, validating everything. Raises ``ValueError``."""
    preset: dict[str, Any] = {}
    if args.preset:
        try:
            preset = json.loads(args.preset.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"cannot read preset {args.preset}: {e}") from e
        if not isinstance(preset, dict):
            raise ValueError(f"preset {args.preset} must be a JSON object")

    model_id = args.model or preset.get("model") or DEFAULT_MODEL
    model_cls = registry.get_class(model_id)
    params: dict[str, Any] = {}
    if preset.get("model") in (None, model_id):
        params.update(preset.get("params") or {})
    elif preset.get("params"):
        log.warning("ignoring preset params for %s (using --model %s)", preset["model"], model_id)
    post_values: dict[str, Any] = dict(preset.get("postprocess") or {})

    model_names = {p.name for p in model_cls.params}
    post_names = {p.name for p in POSTPROCESS_PARAMS}
    model_pairs: dict[str, str] = {}
    post_pairs: dict[str, str] = {}
    for item in args.param:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--param expects K=V, got {item!r}")
        if key in model_names:
            model_pairs[key] = value
        elif key in post_names:
            post_pairs[key] = value
        else:
            raise ValueError(
                f"unknown param {key!r} for {model_id}; model params: {sorted(model_names)}, "
                f"postprocess: {sorted(post_names)}"
            )
    params.update(parse_params(model_cls.params, model_pairs))
    post_values.update(parse_params(POSTPROCESS_PARAMS, post_pairs))
    return Settings(
        model_id, model_cls.validate_params(params), Postprocess.from_mapping(post_values)
    )


def plan_jobs(
    src: Path, dst: Path, fmt: OutputFormat | None, recursive: bool
) -> list[tuple[Path, Path, OutputFormat]]:
    """Map inputs to (input, output, format). Raises ``ValueError`` for a bad INPUT."""
    if src.is_dir():
        pattern = "**/*" if recursive else "*"
        files = sorted(
            f for f in src.glob(pattern) if f.is_file() and f.suffix.lower() in INPUT_SUFFIXES
        )
        jobs = []
        for f in files:
            f_fmt = fmt or _SUFFIX_FORMATS.get(f.suffix.lower(), "png")
            jobs.append((f, (dst / f.relative_to(src)).with_suffix(EXTENSIONS[f_fmt]), f_fmt))
        return jobs
    if not src.is_file():
        raise ValueError(f"input {src} does not exist")
    if dst.is_dir() or not dst.suffix:
        f_fmt = fmt or _SUFFIX_FORMATS.get(src.suffix.lower(), "png")
        return [(src, dst / (src.stem + EXTENSIONS[f_fmt]), f_fmt)]
    out_fmt = fmt or _SUFFIX_FORMATS.get(dst.suffix.lower())
    if out_fmt is None:
        raise ValueError(f"cannot infer format from {dst.name}; use --format")
    return [(src, dst, out_fmt)]


def list_models(registry: Registry) -> None:
    for model_id in registry.ids():
        cls = registry.get_class(model_id)
        log.info("%s  (%s, %s)", model_id, cls.display_name, cls.license)
        for p in cls.params:
            log.info("    %s: %s = %r  %s", p.name, p.kind, p.default, p.help)
    log.info("postprocess (all models):")
    for p in POSTPROCESS_PARAMS:
        log.info("    %s: %s = %r  [%s, %s]", p.name, p.kind, p.default, p.min, p.max)


def main(argv: Sequence[str] | None = None, registry: Registry | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(message)s", stream=sys.stderr)
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    registry = registry or default_registry()

    if args.list_models:
        list_models(registry)
        return 0
    if args.input is None or args.output is None:
        parser.error("INPUT and OUTPUT are required")

    try:
        settings = resolve_settings(args, registry)
        jobs = plan_jobs(args.input, args.output, args.format, args.recursive)
        device = select_device(args.device)
    except ValueError as e:
        log.error("error: %s", e)
        return 2
    if not jobs:
        log.error("error: no images found in %s", args.input)
        return 1

    summary = Summary()
    pending = []
    for job in jobs:
        if job[1].exists() and not args.overwrite:
            log.info("skip %s (exists)", job[1])
            summary.skipped += 1
        else:
            pending.append(job)

    if pending:
        try:
            model = registry.get(settings.model, device)
        except Exception as e:
            log.error("error: cannot load %s: %s", settings.model, e)
            return 1
        for i, (src, dst, fmt) in enumerate(pending, 1):
            try:
                pipeline.colorize_file(
                    src, dst, model, settings.params, settings.post, fmt=fmt, quality=args.quality
                )
            except Exception as e:
                log.error("[%d/%d] FAILED %s: %s", i, len(pending), src, e)
                log.debug("traceback", exc_info=True)
                summary.failed.append(src)
            else:
                log.info("[%d/%d] %s -> %s", i, len(pending), src, dst)
                summary.done += 1

    log.info(
        "%d colorized, %d skipped, %d failed", summary.done, summary.skipped, len(summary.failed)
    )
    for path in summary.failed:
        log.info("  failed: %s", path)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
