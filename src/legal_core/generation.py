"""Grounded answer generation over retrieved legal chunks (issue #7).

Generation is intentionally separate from retrieval. The caller supplies the
retrieved chunks, and this module builds a context containing only those
chunks before invoking the provider.
"""
from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMProvider, get_llm_provider
from .retrieval import RetrievedChunk


class GenerationError(RuntimeError):
    """Raised when the provider fails to produce a usable grounded answer."""


@dataclass(frozen=True)
class GeneratedAnswer:
    question: str
    answer: str
    context_chunk_ids: tuple[str, ...]


class AnswerGenerator:
    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider or get_llm_provider()

    @staticmethod
    def build_context(chunks: list[RetrievedChunk]) -> str:
        """Serialize only retrieved chunks into the provider context."""
        blocks = []
        for chunk in chunks:
            blocks.append(
                f"[SOURCE {chunk.rank}: {chunk.citation}]\n{chunk.text.strip()}"
            )
        return "\n\n".join(blocks)

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> GeneratedAnswer:
        if not question or not question.strip():
            raise ValueError("question must not be blank")
        if not chunks:
            raise ValueError("at least one retrieved chunk is required for grounded generation")

        context = self.build_context(chunks)
        try:
            answer = self.provider.generate(
                question=question.strip(),
                context=context,
            )
        except Exception as exc:  # provider boundary: normalize failures for callers
            raise GenerationError(f"Answer generation failed: {exc}") from exc

        if not isinstance(answer, str) or not answer.strip():
            raise GenerationError("Answer generation failed: provider returned an empty answer")

        return GeneratedAnswer(
            question=question.strip(),
            answer=answer.strip(),
            context_chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        )
