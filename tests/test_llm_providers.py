"""Tests for hosted LLM providers (issue #29); no network is used."""
from __future__ import annotations

import io
import urllib.error

import pytest

from legal_core.llm import (
    AnthropicProvider,
    LLMProviderError,
    LocalExtractiveProvider,
    OpenAIProvider,
    get_llm_provider,
)


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b""))


def make(cls, responses, **kw):
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return cls("k", transport=transport, sleep=lambda s: None, **kw), calls


ANTH_OK = {"content": [{"type": "text", "text": " Answer [SOURCE 1: PPC s.302] "}]}
OAI_OK = {"choices": [{"message": {"content": "Answer [SOURCE 1: PPC s.302]"}}]}


def test_anthropic_request_and_parse():
    p, calls = make(AnthropicProvider, [ANTH_OK])
    assert p.generate(question="q?", context="[SOURCE 1: PPC s.302]\ntext") == "Answer [SOURCE 1: PPC s.302]"
    url, headers, payload, timeout = calls[0]
    assert headers["x-api-key"] == "k" and "SOURCE" in payload["system"]
    assert "q?" in payload["messages"][0]["content"] and timeout == 30.0


def test_openai_request_and_parse():
    p, calls = make(OpenAIProvider, [OAI_OK])
    assert p.generate(question="q?", context="ctx").startswith("Answer")
    assert calls[0][1]["Authorization"] == "Bearer k"


def test_retries_then_succeeds_on_429_and_timeout():
    p, calls = make(OpenAIProvider, [http_error(429), TimeoutError(), OAI_OK])
    assert p.generate(question="q", context="c")
    assert len(calls) == 3


def test_auth_error_not_retried():
    p, calls = make(AnthropicProvider, [http_error(401)])
    with pytest.raises(LLMProviderError, match="authentication"):
        p.generate(question="q", context="c")
    assert len(calls) == 1


def test_gives_up_after_max_retries():
    p, calls = make(AnthropicProvider, [http_error(503)] * 3)
    with pytest.raises(LLMProviderError, match="503"):
        p.generate(question="q", context="c")
    assert len(calls) == 3


def test_bad_shape_and_empty_answer():
    p, _ = make(OpenAIProvider, [{"oops": 1}])
    with pytest.raises(LLMProviderError, match="unexpected"):
        p.generate(question="q", context="c")
    p, _ = make(OpenAIProvider, [{"choices": [{"message": {"content": "  "}}]}])
    with pytest.raises(LLMProviderError, match="empty"):
        p.generate(question="q", context="c")


def test_factory_from_env():
    assert isinstance(get_llm_provider(env={}), LocalExtractiveProvider)
    p = get_llm_provider(env={"LLM_PROVIDER": "anthropic", "LLM_API_KEY": "k", "LLM_MODEL": "m"})
    assert isinstance(p, AnthropicProvider) and p.model == "m"
    p = get_llm_provider("local-openai", env={"LLM_BASE_URL": "http://localhost:8000/v1/chat/completions"})
    assert isinstance(p, OpenAIProvider) and p.url.startswith("http://localhost")
    with pytest.raises(ValueError):
        get_llm_provider("anthropic", env={})
    with pytest.raises(ValueError):
        get_llm_provider("bogus", env={})
