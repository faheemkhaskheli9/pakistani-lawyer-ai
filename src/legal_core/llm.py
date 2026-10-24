"""Swappable language-model provider interface.

The default provider is deliberately local and deterministic so the project
runs offline with no API key. Hosted providers can implement the same small
interface without changing the legal answer-generation layer.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

SYSTEM_PROMPT = (
    "You are a legal research assistant for Pakistani law. Answer ONLY from the "
    "numbered sources in the supplied context. After every legal claim cite the "
    "source it comes from using its exact label, e.g. [SOURCE 1: <citation>]. "
    "If the sources do not answer the question, say so plainly and do not guess. "
    "Never invent sections, case names or citations. This is legal information, "
    "not legal advice."
)


def build_user_prompt(*, question: str, context: str) -> str:
    return f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"


class LLMProviderError(RuntimeError):
    """Mapped provider failure (auth, rate limit, timeout, server error)."""


class LLMProvider(ABC):
    """Minimal provider boundary used by grounded legal answer generation."""

    @abstractmethod
    def generate(self, *, question: str, context: str) -> str:
        """Generate an answer using only the supplied grounded context."""


class LocalExtractiveProvider(LLMProvider):
    """Offline fallback that returns a concise extract from supplied context.

    This is not intended to imitate a full LLM. It makes the answer pipeline
    runnable, deterministic and testable without network access.
    """

    def generate(self, *, question: str, context: str) -> str:
        del question  # the local fallback is intentionally context-only
        normalized = " ".join(context.split())
        if not normalized:
            raise ValueError("grounded context is empty")
        limit = 900
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."


Transport = Callable[[str, dict, dict, float], dict]


def _http_post_json(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


class _HostedProvider(LLMProvider):
    """Shared timeout/retry/error-mapping logic for HTTP chat providers."""

    default_model = ""
    default_url = ""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise ValueError("LLM_API_KEY is required for hosted LLM providers")
        self.api_key = api_key
        self.model = model or self.default_model
        self.url = base_url or self.default_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = transport or _http_post_json
        self._sleep = sleep

    def _request(self, *, question: str, context: str) -> tuple[dict, dict]:
        raise NotImplementedError

    def _extract(self, body: dict) -> str:
        raise NotImplementedError

    def generate(self, *, question: str, context: str) -> str:
        if not context.strip():
            raise ValueError("grounded context is empty")
        headers, payload = self._request(question=question, context=context)
        for attempt in range(self.max_retries + 1):
            try:
                body = self._transport(self.url, headers, payload, self.timeout)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise LLMProviderError("LLM authentication failed") from exc
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise LLMProviderError(f"LLM request failed with HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == self.max_retries:
                    raise LLMProviderError(f"LLM request failed: {exc}") from exc
            self._sleep(2**attempt)
        try:
            text = self._extract(body)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("LLM returned an unexpected response shape") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMProviderError("LLM returned an empty answer")
        return text.strip()


class AnthropicProvider(_HostedProvider):
    default_model = "claude-sonnet-5-5"
    default_url = "https://api.anthropic.com/v1/messages"

    def _request(self, *, question: str, context: str) -> tuple[dict, dict]:
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": build_user_prompt(question=question, context=context)}
            ],
        }
        return headers, payload

    def _extract(self, body: dict) -> str:
        return "".join(
            block["text"] for block in body["content"] if block.get("type") == "text"
        )


class OpenAIProvider(_HostedProvider):
    default_model = "gpt-4o-mini"
    default_url = "https://api.openai.com/v1/chat/completions"

    def _request(self, *, question: str, context: str) -> tuple[dict, dict]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question=question, context=context)},
            ],
        }
        return headers, payload

    def _extract(self, body: dict) -> str:
        return body["choices"][0]["message"]["content"]


def get_llm_provider(
    name: str | None = None, env: Mapping[str, str] | None = None
) -> LLMProvider:
    """Return the configured provider (``LLM_PROVIDER``, default offline).

    Hosted providers read ``LLM_API_KEY``, optional ``LLM_MODEL`` and
    ``LLM_BASE_URL`` (the latter enables OpenAI-compatible local servers).
    """
    env = os.environ if env is None else env
    provider = (name or env.get("LLM_PROVIDER") or "local").strip().lower()
    if provider in {"local", "fake", "extractive"}:
        return LocalExtractiveProvider()
    if provider in {"anthropic", "openai", "openai-compatible", "local-openai"}:
        cls = AnthropicProvider if provider == "anthropic" else OpenAIProvider
        api_key = env.get("LLM_API_KEY", "")
        base_url = env.get("LLM_BASE_URL") or None
        if provider == "local-openai" and not api_key:
            api_key = "not-needed"  # local servers typically ignore auth
        return cls(api_key, env.get("LLM_MODEL") or None, base_url=base_url)
    raise ValueError(f"Unsupported LLM provider: {name!r}")
