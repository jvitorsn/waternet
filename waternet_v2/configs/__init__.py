"""Configuration utilities for WaterNet v2."""

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration, merging with defaults.

    Args:
        path: Optional path to a custom YAML config file.  When None the
            bundled default.yaml is loaded unchanged.

    Returns:
        Nested dictionary of configuration values.
    """
    with open(_DEFAULT_CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)

    if path is not None:
        with open(path) as fh:
            overrides = yaml.safe_load(fh) or {}
        _deep_merge(config, overrides)

    return config


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
