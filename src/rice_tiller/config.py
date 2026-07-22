from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from rice_tiller.exceptions import ConfigurationError


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    project_root = _find_project_root(config_path.parent)
    data["_config_path"] = str(config_path)
    data["_project_root"] = str(project_root)
    validate_config(data)
    return data


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    _deep_merge(merged, override)
    return merged


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def validate_config(config: dict[str, Any]) -> None:
    required = ["model", "inference", "height", "tiller_count"]
    missing = [name for name in required if name not in config]
    if missing:
        raise ConfigurationError(f"Missing configuration sections: {', '.join(missing)}")

    vertical_axis = str(config.get("input", {}).get("vertical_axis", "z")).lower()
    if vertical_axis != "z":
        raise ConfigurationError(
            "Version 0.1 requires CloudCompare input aligned to the positive Z axis"
        )

    inference = config["inference"]
    if int(inference.get("chunk_size", 0)) <= 1:
        raise ConfigurationError("inference.chunk_size must be greater than 1")
    if int(inference.get("voting_passes", 0)) < 1:
        raise ConfigurationError("inference.voting_passes must be at least 1")

    raster = config["tiller_count"].get("raster", {})
    if int(raster.get("rows", 0)) < 2 or int(raster.get("cols", 0)) < 2:
        raise ConfigurationError("CHM raster must have at least 2 rows and columns")


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
