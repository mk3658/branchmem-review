"""Content-hash cache for LLM calls: never re-spend on a repeated identical prompt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class LLMCache:
    """Directory-of-JSON-files cache, keyed by sha256 of the full call parameters.

    One file per unique (backend, model, system, prompt, decoding params) tuple,
    so re-running analysis against a cached run costs nothing.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(**kwargs: Any) -> str:
        payload = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set(self, key: str, value: dict) -> None:
        self._path(key).write_text(json.dumps(value, indent=2))

    def __contains__(self, key: str) -> bool:
        return self._path(key).exists()

    def size(self) -> int:
        return sum(1 for _ in self.cache_dir.glob("*.json"))
