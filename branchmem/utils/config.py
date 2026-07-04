"""YAML config loading with simple dict->attribute access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(*paths: str | Path) -> Config:
    """Load one or more YAML files, later files override earlier ones (shallow merge per top key)."""
    merged: dict[str, Any] = {}
    for path in paths:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return Config(merged)
