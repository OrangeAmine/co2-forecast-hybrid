"""Configuration loading utilities for YAML-based hyperparameter management."""

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary whose values take precedence.

    Returns:
        Merged dictionary with override values taking precedence.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_paths: list[str | Path]) -> dict[str, Any]:
    """Load and merge multiple YAML config files.

    Later files override earlier ones for duplicate keys. Nested
    dictionaries are merged recursively.

    Args:
        config_paths: List of paths to YAML config files.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If a config file does not exist.
    """
    merged: dict[str, Any] = {}
    for path in config_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, config)
    return merged
