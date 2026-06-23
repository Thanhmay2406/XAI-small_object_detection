"""Configuration helpers for phase scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    loaded["_config_path"] = str(config_path)
    return loaded


def ensure_file(path: str | Path, description: str) -> Path:
    """Ensure a file exists and return its resolved path."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a file: {resolved}")
    return resolved


def ensure_dir(path: str | Path, description: str, create: bool = False) -> Path:
    """Ensure a directory exists and return its resolved path."""

    resolved = Path(path).expanduser().resolve()
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} is not a directory: {resolved}")
    return resolved


def dump_yaml_config(data: dict[str, Any], path: str | Path) -> Path:
    """Write a YAML config to disk."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return resolved
