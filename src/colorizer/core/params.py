"""Parameter schema shared by models, postprocessing, the UI and the CLI."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

_N = TypeVar("_N", int, float)

ParamKind = Literal["int", "float", "bool", "choice", "str", "seed", "points"]

# A colour hint: pixel coordinates in the original image plus an sRGB colour (0-255).
Point = tuple[int, int, tuple[int, int, int]]

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Param:
    name: str
    kind: ParamKind
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    help: str = ""

    def __post_init__(self) -> None:
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"param {self.name!r}: kind 'choice' requires choices")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"param {self.name!r}: min > max")
        # The default must itself be valid; this catches schema typos at import time.
        self.validate(self.default)

    def validate(self, value: Any) -> Any:
        """Return ``value`` normalised to this param's type, or raise ``ValueError``."""
        match self.kind:
            case "int" | "seed":
                return self._check_range(_as_int(self.name, value))
            case "float":
                return self._check_range(_as_float(self.name, value))
            case "bool":
                if not isinstance(value, bool):
                    raise ValueError(f"param {self.name!r}: expected bool, got {value!r}")
                return value
            case "choice":
                assert self.choices is not None
                if value not in self.choices:
                    raise ValueError(
                        f"param {self.name!r}: {value!r} not one of {list(self.choices)}"
                    )
                return value
            case "str":
                if not isinstance(value, str):
                    raise ValueError(f"param {self.name!r}: expected str, got {value!r}")
                return value
            case "points":
                return _as_points(self.name, value)
        raise AssertionError(f"unhandled kind {self.kind}")  # pragma: no cover

    def parse(self, text: str) -> Any:
        """Parse a string (CLI ``k=v``) into a validated value."""
        match self.kind:
            case "int" | "seed":
                try:
                    return self.validate(int(text))
                except ValueError as e:
                    raise ValueError(f"param {self.name!r}: invalid integer {text!r}") from e
            case "float":
                try:
                    return self.validate(float(text))
                except ValueError as e:
                    raise ValueError(f"param {self.name!r}: invalid number {text!r}") from e
            case "bool":
                low = text.strip().lower()
                if low in _TRUE:
                    return True
                if low in _FALSE:
                    return False
                raise ValueError(f"param {self.name!r}: invalid bool {text!r}")
            case "points":
                try:
                    return self.validate(json.loads(text))
                except json.JSONDecodeError as e:
                    raise ValueError(f"param {self.name!r}: points must be JSON") from e
            case _:
                return self.validate(text)

    def _check_range(self, value: _N) -> _N:
        if self.min is not None and value < self.min:
            raise ValueError(f"param {self.name!r}: {value} < min {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"param {self.name!r}: {value} > max {self.max}")
        if self.step is not None and self.kind == "int":
            base = self.min if self.min is not None else 0
            if (value - base) % self.step != 0:
                raise ValueError(
                    f"param {self.name!r}: {value} is not {base} + a multiple of {self.step}"
                )
        return value


def _as_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"param {name!r}: expected int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"param {name!r}: expected int, got {value!r}")


def _as_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"param {name!r}: expected number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"param {name!r}: must be finite")
    return result


def _as_points(name: str, value: Any) -> list[Point]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"param {name!r}: expected a list of [x, y, [r, g, b]]")
    points: list[Point] = []
    for item in value:
        try:
            x, y, rgb = item
            r, g, b = rgb
            coords = (_as_int(name, x), _as_int(name, y))
            color = (_as_int(name, r), _as_int(name, g), _as_int(name, b))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"param {name!r}: bad point {item!r}, expected [x, y, [r, g, b]]"
            ) from e
        if min(coords) < 0:
            raise ValueError(f"param {name!r}: negative coordinates in {item!r}")
        if not all(0 <= c <= 255 for c in color):
            raise ValueError(f"param {name!r}: colour out of 0-255 in {item!r}")
        points.append((coords[0], coords[1], color))
    return points


def validate_params(schema: Sequence[Param], values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``values`` against ``schema`` and fill in defaults.

    Raises ``ValueError`` for unknown names or invalid values.
    """
    by_name = {p.name: p for p in schema}
    unknown = sorted(set(values) - set(by_name))
    if unknown:
        raise ValueError(f"unknown param(s) {unknown}; expected one of {sorted(by_name)}")
    return {p.name: p.validate(values[p.name]) if p.name in values else p.default for p in schema}


def parse_params(schema: Sequence[Param], pairs: Mapping[str, str]) -> dict[str, Any]:
    """Parse string values (from the CLI) for the params named in ``pairs``."""
    by_name = {p.name: p for p in schema}
    unknown = sorted(set(pairs) - set(by_name))
    if unknown:
        raise ValueError(f"unknown param(s) {unknown}; expected one of {sorted(by_name)}")
    return {k: by_name[k].parse(v) for k, v in pairs.items()}
