"""OpenAI-compatible chat completion backend (OpenAI proper, or any
OpenAI-API-compatible endpoint via base_url). Reads OPENAI_API_KEY from the
environment — never hardcode keys."""

from __future__ import annotations

import os
from typing import Optional

from branchmem.llm.base import LLMBackend, LLMResponse


class OpenAICompatibleBackend(LLMBackend):
    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        cache=None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, cache=cache, temperature=temperature, max_tokens=max_tokens)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before using the "
                "'openai_compatible' backend, or set llm.backend: mock in the "
                "config for development without spending on real API calls."
            )
        import openai  # local import: keep the dependency optional for mock-only usage

        self._client = openai.OpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        # Learned per-instance after the first call: avoids paying a wasted
        # round-trip on every single request once we know this model/endpoint
        # needs max_completion_tokens and/or rejects a non-default temperature.
        self._use_max_completion_tokens = False
        self._omit_temperature = False

    def _call(self, prompt: str, system: Optional[str]) -> LLMResponse:
        import openai

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = dict(model=self.model, messages=messages)
        if not self._omit_temperature:
            kwargs["temperature"] = self.temperature
        if self._use_max_completion_tokens:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens

        # Newer OpenAI models (o1/o3/gpt-5.x reasoning family) reject
        # max_tokens (want max_completion_tokens) and/or a non-default
        # temperature. Retry with the offending param patched rather than
        # hardcoding per-model quirks, since these vary by provider/model and
        # change over time; remember the fix so later calls skip straight to it.
        resp = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except openai.BadRequestError as exc:
                if attempt == 2:
                    raise
                message = str(exc)
                if "max_tokens" in message and "max_completion_tokens" in message:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    self._use_max_completion_tokens = True
                elif "temperature" in message:
                    kwargs.pop("temperature", None)
                    self._omit_temperature = True
                else:
                    raise

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResponse(
            text=text,
            model=resp.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw=resp.model_dump(),
        )
