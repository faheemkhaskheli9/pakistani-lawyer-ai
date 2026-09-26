"""Swappable language-model provider interface.

The default provider is deliberately local and deterministic so the project
runs offline with no API key. Hosted providers can implement the same small
interface without changing the legal answer-generation layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


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


def get_llm_provider(name: str | None = None) -> LLMProvider:
    """Return the configured provider.

    Only the offline provider is built in today. Additional providers should
    be added behind this function rather than imported by the generation layer.
    """
    provider = (name or "local").strip().lower()
    if provider in {"local", "fake", "extractive"}:
        return LocalExtractiveProvider()
    raise ValueError(f"Unsupported LLM provider: {name!r}")
