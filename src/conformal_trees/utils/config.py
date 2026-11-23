"""YAML config loading. No experiment parameter should be hard-coded in code."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# src/conformal_trees/utils/config.py -> project root is 3 levels above utils/.
# If the package is ever installed non-editable, set CT_PROJECT_ROOT instead.
PROJECT_ROOT = Path(os.environ.get("CT_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(*names: str) -> dict[str, Any]:
    """Load and merge config files (paths relative to configs/), later ones win.

    >>> cfg = load_config("base.yaml")
    """
    merged: dict[str, Any] = {}
    for name in names:
        merged = deep_merge(merged, load_yaml(CONFIG_DIR / name))
    return merged


def load_dataset_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / "datasets" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No dataset config for '{name}' (expected {path})")
    return load_yaml(path)


def resolve_path(path: str | Path) -> Path:
    """Resolve a path from a config file relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
