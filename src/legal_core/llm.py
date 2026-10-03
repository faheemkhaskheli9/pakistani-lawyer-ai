"""Swappable language-model provider interface.

The default provider is deliberately local and deterministic so the project
runs offline with no API key. Hosted providers can implement the same small
interface without changing the legal answer-generation layer.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

GROUNDED_SYSTEM_PROMPT = (
    "You are a legal-information assistant for Pakistani law. Answer the question "
    "using ONLY the numbered sources supplied in the context. Cite each claim inline "
    "with its source label, for example [SOURCE 1: <citation>]. Never cite a source that "
    "was not supplied and never invent sections, case names or citations. If the sources "
    "do not answer the question, reply exactly: INSUFFICIENT_SOURCES. This is legal "
    "information, not legal advice."
)
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"


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


class AnthropicProvider(LLMProvider):
    """Hosted provider backed by the Anthropic Messages API.

    The `anthropic` package is imported lazily so the offline default never
    needs it. A pre-built client may be injected for testing.
    """

    def __init__(self, *, model: str | None = None, api_key: str | None = None,
                 max_tokens: int = 1024, client=None):
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
        self.max_tokens = max_tokens
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "AnthropicProvider requires the 'anthropic' package: "
                    "pip install 'pakistani-lawyer-ai[anthropic]'"
                ) from exc
            key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY (or LLM_API_KEY) is required for the anthropic provider")
            client = anthropic.Anthropic(api_key=key)
        self._client = client

    def generate(self, *, question: str, context: str) -> str:
        if not context.strip():
            raise ValueError("grounded context is empty")
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=GROUNDED_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}",
                }
            ],
        )
        text = "".join(
            getattr(block, "text", "") for block in getattr(message, "content", [])
        ).strip()
        if not text or text == "INSUFFICIENT_SOURCES":
            raise ValueError("model found the supplied sources insufficient to answer")
        return text


def get_llm_provider(name: str | None = None) -> LLMProvider:
    """Return the configured provider.

    `name` overrides `$LLM_PROVIDER` (default: local). Providers are added
    behind this function rather than imported by the generation layer.
    """
    provider = (name or os.environ.get("LLM_PROVIDER") or "local").strip().lower()
    if provider == "anthropic":
        return AnthropicProvider()
    if provider in {"local", "fake", "extractive"}:
        return LocalExtractiveProvider()
    raise ValueError(f"Unsupported LLM provider: {name!r}")
