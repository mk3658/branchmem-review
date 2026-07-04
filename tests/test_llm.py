import os
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from branchmem.llm.base import build_backend
from branchmem.llm.cache import LLMCache
from branchmem.llm.mock_backend import MockBackend
from branchmem.llm.openai_compatible_backend import OpenAICompatibleBackend


def test_cache_roundtrip(tmp_path):
    cache = LLMCache(tmp_path / "cache")
    key = LLMCache.make_key(backend="mock", model="m", system="", prompt="hi", temperature=0.0, max_tokens=10)
    assert cache.get(key) is None
    cache.set(key, {"text": "hello", "model": "m", "input_tokens": 1, "output_tokens": 1, "raw": {}})
    assert cache.get(key)["text"] == "hello"
    assert key in cache
    assert cache.size() == 1


def test_mock_backend_call_is_cached(tmp_path):
    cache = LLMCache(tmp_path / "cache")
    backend = MockBackend(cache=cache)

    r1 = backend.complete("What is the capital of France?")
    assert r1.cached is False
    assert cache.size() == 1

    r2 = backend.complete("What is the capital of France?")
    assert r2.cached is True
    assert r2.text == r1.text
    assert cache.size() == 1  # no new entry for a repeated identical call


def test_different_prompts_produce_different_cache_entries(tmp_path):
    cache = LLMCache(tmp_path / "cache")
    backend = MockBackend(cache=cache)
    backend.complete("prompt A")
    backend.complete("prompt B")
    assert cache.size() == 2


def test_mock_backend_canned_responses():
    backend = MockBackend(canned_responses={"CONFLICT": "yes, this is a conflict"})
    resp = backend.complete("Does this pair CONFLICT?")
    assert resp.text == "yes, this is a conflict"


def test_build_backend_mock(tmp_path):
    backend = build_backend({"backend": "mock", "model": "mock-1", "cache_dir": str(tmp_path / "cache")})
    assert isinstance(backend, MockBackend)
    resp = backend.complete("hello")
    assert resp.text.startswith("[mock-response-")


def test_anthropic_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_backend({"backend": "anthropic", "model": "claude-fable-5"})


def test_openai_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_backend({"backend": "openai_compatible", "model": "gpt-x"})


def _bad_request_error(message: str) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": message}})
    return openai.BadRequestError(message, response=response, body={"error": {"message": message}})


def test_openai_backend_retries_with_max_completion_tokens(monkeypatch):
    # Newer OpenAI models (o1/o3/gpt-5.x) reject max_tokens and require
    # max_completion_tokens instead — discovered when running the real Phase 5
    # pilot against gpt-5.4-nano. This test locks in the retry behavior.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = OpenAICompatibleBackend(model="gpt-5.4-nano", max_tokens=50)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
    mock_response.model = "gpt-5.4-nano"
    mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=1)
    mock_response.model_dump.return_value = {}

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if "max_tokens" in kwargs:
            raise _bad_request_error(
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."
            )
        return mock_response

    backend._client.chat.completions.create = fake_create
    response = backend._call("hello", None)
    assert response.text == "OK"
    assert len(calls) == 2
    assert "max_tokens" in calls[0]
    assert "max_completion_tokens" in calls[1]


def test_openai_backend_retries_by_dropping_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = OpenAICompatibleBackend(model="o1", temperature=0.0)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
    mock_response.model = "o1"
    mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=1)
    mock_response.model_dump.return_value = {}

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise _bad_request_error("Unsupported value: 'temperature' does not support 0.0 with this model.")
        return mock_response

    backend._client.chat.completions.create = fake_create
    response = backend._call("hello", None)
    assert response.text == "OK"
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
