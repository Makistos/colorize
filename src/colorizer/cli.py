"""Batch command-line interface.

colorizer INPUT OUTPUT [--model ID] [--param k=v ...] [--preset FILE] [--device NAME]
          [--restore ID[,ID...]] [--rparam ID.KEY=VALUE ...]
          [--format png|jpg|tiff] [-r] [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colorizer.core.batch import BatchItem, run_batch
from colorizer.core.params import parse_params
from colorizer.core.pipeline import EXTENSIONS, INPUT_SUFFIXES, SUFFIX_FORMATS, OutputFormat
from colorizer.core.postprocess import POSTPROCESS_PARAMS, Postprocess
from colorizer.core.presets import load_preset
from colorizer.core.registry import Registry, default_registry
from colorizer.core.restore import RESTORE_PARAMS, RestoreStep, validate_restore_params
from colorizer.core.runtime import select_device

log = logging.getLogger("colorizer")

DEFAULT_MODEL = "ddcolor"


@dataclass
class Settings:
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    post: Postprocess = field(default_factory=Postprocess)
    restore: list[RestoreStep] = field(default_factory=list)


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
    p.add_argument(
        "--preset", help="preset name (from the UI) or JSON file {model, params, postprocess}"
    )
    p.add_argument(
        "--restore",
        metavar="ID[,ID...]",
        help="restorers to run before colorizing, in order (overrides the preset's)",
    )
    p.add_argument(
        "--rparam",
        action="append",
        default=[],
        metavar="ID.KEY=V",
        help="restorer parameter, e.g. realesrgan.scale=4 or codeformer.blend=0.5; repeatable",
    )
    p.add_argument("--device", help="auto, cpu, cuda, rocm, directml, coreml")
    p.add_argument("--format", choices=list(EXTENSIONS), help="output format")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    p.add_argument("-r", "--recursive", action="store_true", help="recurse into directories")
    p.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    p.add_argument(
        "--list-models", action="store_true", help="list models, restorers and their params"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def resolve_settings(args: argparse.Namespace, registry: Registry) -> Settings:
    """Merge preset and CLI arguments, validating everything. Raises ``ValueError``."""
    preset = load_preset(args.preset) if args.preset else None

    model_id = args.model or (preset.model if preset else DEFAULT_MODEL)
    model_cls = registry.get_class(model_id)
    params: dict[str, Any] = {}
    if preset and preset.model == model_id:
        params.update(preset.params)
    elif preset and preset.params:
        log.warning("ignoring preset params for %s (using --model %s)", preset.model, model_id)
    post_values: dict[str, Any] = dict(preset.postprocess) if preset else {}

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
        model_id,
        model_cls.validate_params(params),
        Postprocess.from_mapping(post_values),
        resolve_restore(args, registry, preset.restore if preset else ()),
    )


def resolve_restore(
    args: argparse.Namespace, registry: Registry, preset_steps: Sequence[RestoreStep]
) -> list[RestoreStep]:
    """The restore chain from ``--restore``/``--rparam`` (or the preset), validated."""
    if args.restore is not None:
        ids = [i.strip() for i in args.restore.split(",") if i.strip()]
        steps = [RestoreStep(i, {}) for i in ids]
    else:
        steps = [RestoreStep(s.id, dict(s.params)) for s in preset_steps]
    values: dict[str, dict[str, Any]] = {s.id: dict(s.params) for s in steps}
    for item in args.rparam:
        target, sep, value = item.partition("=")
        rid, dot, key = target.strip().partition(".")
        if not sep or not dot or not rid or not key:
            raise ValueError(f"--rparam expects ID.KEY=V, got {item!r}")
        if rid not in values:
            raise ValueError(
                f"--rparam {item!r}: {rid!r} is not in the restore chain {list(values)}"
            )
        schema = (*registry.restorers.get_class(rid).params, *RESTORE_PARAMS)
        values[rid].update(parse_params(schema, {key: value}))
    # Same id twice in a chain shares its params.
    return [
        RestoreStep(s.id, validate_restore_params(registry.restorers.get_class(s.id), values[s.id]))
        for s in steps
    ]


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
            f_fmt = fmt or SUFFIX_FORMATS.get(f.suffix.lower(), "png")
            jobs.append((f, (dst / f.relative_to(src)).with_suffix(EXTENSIONS[f_fmt]), f_fmt))
        return jobs
    if not src.is_file():
        raise ValueError(f"input {src} does not exist")
    if dst.is_dir() or not dst.suffix:
        f_fmt = fmt or SUFFIX_FORMATS.get(src.suffix.lower(), "png")
        return [(src, dst / (src.stem + EXTENSIONS[f_fmt]), f_fmt)]
    out_fmt = fmt or SUFFIX_FORMATS.get(dst.suffix.lower())
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
    log.info("restorers (--restore ID[,ID...], --rparam ID.KEY=V):")
    for restorer_id in registry.restorers.ids():
        rcls = registry.restorers.get_class(restorer_id)
        state = "" if rcls.enabled else "  [not implemented yet]"
        log.info("%s  (%s, %s)%s", restorer_id, rcls.display_name, rcls.license, state)
        for p in rcls.params:
            log.info("    %s: %s = %r  %s", p.name, p.kind, p.default, p.help)
    log.info("restore params (all restorers):")
    for p in RESTORE_PARAMS:
        log.info("    %s: %s = %r  %s", p.name, p.kind, p.default, p.help)


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

    for step in settings.restore:
        warning = registry.restorers.get_class(step.id).warning
        if warning:
            log.warning("warning: %s", warning)
    items = [BatchItem(src, dst, fmt) for src, dst, fmt in jobs]
    try:
        result = run_batch(
            items,
            lambda: registry.get(settings.model, device),
            settings.params,
            settings.post,
            quality=args.quality,
            overwrite=args.overwrite,
            restore=settings.restore,
            get_restorer=lambda rid: registry.restorers.get(rid, device),
        )
    except Exception as e:  # per-file errors are collected; this is a load failure
        what = ", ".join([settings.model, *(s.id for s in settings.restore)])
        log.error("error: cannot load %s: %s", what, e)
        return 1
    log.info("%s", result.summary())
    for item, _ in result.failed:
        log.info("  failed: %s", item.src)
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
