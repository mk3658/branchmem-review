"""Deterministic, offline LLM backend for unit tests and development only.

NEVER use this backend to produce reported experimental results (Phase 5/6) —
see PROGRESS.md and the project's determinism/honesty requirement. Its output
is a hash-derived placeholder string, not a model's actual reasoning.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from branchmem.llm.base import LLMBackend, LLMResponse


class MockBackend(LLMBackend):
    name = "mock"

    def __init__(
        self,
        model: str = "mock-1",
        cache=None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        canned_responses: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(model=model, cache=cache, temperature=temperature, max_tokens=max_tokens)
        # substring-triggered canned responses let unit tests exercise specific
        # code paths (e.g. merge/detector logic) deterministically.
        self.canned_responses = canned_responses or {}

    def _call(self, prompt: str, system: Optional[str]) -> LLMResponse:
        for trigger, response_text in self.canned_responses.items():
            if trigger in prompt:
                return LLMResponse(
                    text=response_text,
                    model=self.model,
                    input_tokens=len(prompt.split()),
                    output_tokens=len(response_text.split()),
                )
        digest = hashlib.sha256(((system or "") + prompt).encode("utf-8")).hexdigest()[:8]
        text = f"[mock-response-{digest}]"
        return LLMResponse(
            text=text, model=self.model, input_tokens=len(prompt.split()), output_tokens=len(text.split())
        )
