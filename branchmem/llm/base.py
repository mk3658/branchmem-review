"""LLM backend interface: every real call is cached and logged (prompt + response)
for auditability and later qualitative analysis, per the project's cost-control and
reproducibility requirements.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from branchmem.llm.cache import LLMCache
from branchmem.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    """Subclasses implement `_call`; caching, logging, and the public API live here."""

    name: str = "base"

    def __init__(
        self,
        model: str,
        cache: Optional[LLMCache] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache

    @abstractmethod
    def _call(self, prompt: str, system: Optional[str]) -> LLMResponse:
        """Perform the actual network/inference call. No caching logic here."""

    def complete(self, prompt: str, system: Optional[str] = None, use_cache: bool = True) -> LLMResponse:
        cache_key = None
        if self.cache is not None and use_cache:
            cache_key = LLMCache.make_key(
                backend=self.name,
                model=self.model,
                system=system or "",
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info("cache HIT [%s] %s (%s...)", self.name, self.model, cache_key[:10])
                return LLMResponse(
                    text=cached["text"],
                    model=cached["model"],
                    input_tokens=cached["input_tokens"],
                    output_tokens=cached["output_tokens"],
                    cached=True,
                    raw=cached.get("raw", {}),
                )

        t0 = time.time()
        response = self._call(prompt, system)
        elapsed = time.time() - t0
        logger.info(
            "cache MISS [%s] %s: %d in / %d out tokens, %.2fs",
            self.name, self.model, response.input_tokens, response.output_tokens, elapsed,
        )

        if cache_key is not None:
            self.cache.set(
                cache_key,
                {
                    "text": response.text,
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "raw": response.raw,
                    "prompt": prompt,
                    "system": system or "",
                },
            )
        return response


def build_backend(llm_config: dict, cache_dir_override: Optional[str] = None) -> LLMBackend:
    """Construct a backend from a config dict (e.g. loaded from configs/*.yaml)."""
    from branchmem.llm.anthropic_backend import AnthropicBackend
    from branchmem.llm.mock_backend import MockBackend
    from branchmem.llm.openai_compatible_backend import OpenAICompatibleBackend

    backend_name = llm_config["backend"]
    cache_dir = cache_dir_override or llm_config.get("cache_dir", "llm_cache")
    cache = LLMCache(cache_dir) if backend_name != "mock" else LLMCache(cache_dir)

    common = dict(
        model=llm_config["model"],
        cache=cache,
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=llm_config.get("max_tokens", 1024),
    )
    if backend_name == "mock":
        return MockBackend(**common)
    if backend_name == "anthropic":
        return AnthropicBackend(**common)
    if backend_name == "openai_compatible":
        return OpenAICompatibleBackend(**common, base_url=llm_config.get("base_url"))
    raise ValueError(f"Unknown llm.backend: {backend_name!r}")
