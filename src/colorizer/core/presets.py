"""Named presets: ``{model, params, postprocess}`` JSON files in the config directory."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colorizer.core.postprocess import Postprocess
from colorizer.core.registry import Registry

_NAME = re.compile(r"^[\w][\w .()+-]{0,79}$")


def config_dir() -> Path:
    """``$XDG_CONFIG_HOME/colorizer`` or ``~/.config/colorizer``."""
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "colorizer"


def presets_dir() -> Path:
    return config_dir() / "presets"


@dataclass(frozen=True)
class Preset:
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    postprocess: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Any) -> Preset:
        """Check the JSON structure; raises ``ValueError``. Values are checked by ``validate``."""
        if not isinstance(data, Mapping):
            raise ValueError("preset must be a JSON object")
        model = data.get("model")
        params = data.get("params", {})
        post = data.get("postprocess", {})
        params = {} if params is None else params
        post = {} if post is None else post
        if not isinstance(model, str) or not model:
            raise ValueError("preset needs a 'model' string")
        if not isinstance(params, Mapping) or not isinstance(post, Mapping):
            raise ValueError("preset 'params' and 'postprocess' must be objects")
        return cls(model, dict(params), dict(post))

    def validate(self, registry: Registry) -> Preset:
        """Return a copy with all values validated and defaults filled; raises ``ValueError``."""
        model_cls = registry.get_class(self.model)
        return Preset(
            self.model,
            model_cls.validate_params(self.params),
            Postprocess.from_mapping(self.postprocess).to_dict(),
        )

    def to_json(self) -> str:
        data = {"model": self.model, "params": self.params, "postprocess": self.postprocess}
        return json.dumps(data, indent=2, sort_keys=False) + "\n"


def check_name(name: str) -> str:
    """Preset names become file names; reject paths and odd characters."""
    name = name.strip()
    if not _NAME.match(name):
        raise ValueError(
            f"invalid preset name {name!r}: use letters, digits, spaces and . _ - ( ) +"
        )
    return name


def list_presets(directory: Path | None = None) -> list[str]:
    directory = directory or presets_dir()
    if not directory.is_dir():
        return []
    return sorted((p.stem for p in directory.glob("*.json")), key=str.lower)


def preset_path(name: str, directory: Path | None = None) -> Path:
    return (directory or presets_dir()) / f"{check_name(name)}.json"


def load_preset(name_or_path: str | Path, directory: Path | None = None) -> Preset:
    """Load by preset name, or from a file path. Raises ``ValueError`` on any problem."""
    candidate = Path(name_or_path).expanduser()
    is_path = candidate.suffix == ".json" or len(candidate.parts) > 1
    path = candidate if is_path else preset_path(str(name_or_path), directory)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"preset {name_or_path!s} not found ({path})") from None
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read preset {path}: {e}") from e
    return Preset.from_mapping(data)


def save_preset(name: str, preset: Preset, directory: Path | None = None) -> Path:
    path = preset_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(preset.to_json(), encoding="utf-8")
    return path


def delete_preset(name: str, directory: Path | None = None) -> None:
    preset_path(name, directory).unlink(missing_ok=True)
