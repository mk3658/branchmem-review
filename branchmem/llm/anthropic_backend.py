"""Anthropic API backend. Reads ANTHROPIC_API_KEY from the environment — never
hardcode keys. Fails with a clear error (not a silent fallback) if the key is
missing; callers who want to run without a key should select the mock backend."""

from __future__ import annotations

import os
from typing import Optional

from branchmem.llm.base import LLMBackend, LLMResponse


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        cache=None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, cache=cache, temperature=temperature, max_tokens=max_tokens)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before using the 'anthropic' "
                "backend, or set llm.backend: mock in the config for development "
                "without spending on real API calls."
            )
        import anthropic  # local import: keep the dependency optional for mock-only usage

        self._client = anthropic.Anthropic(api_key=key)

    def _call(self, prompt: str, system: Optional[str]) -> LLMResponse:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw=resp.model_dump(),
        )
