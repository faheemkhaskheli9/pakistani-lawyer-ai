"""Question-answer response models with explicit source citations (issue #8)."""
from __future__ import annotations

from dataclasses import dataclass

from .generation import AnswerGenerator, GeneratedAnswer
from .retrieval import RetrievedChunk


@dataclass(frozen=True)
class AnswerCitation:
    chunk_id: str
    source: str
    section: str | None
    citation: str
    score: float
    excerpt: str


@dataclass(frozen=True)
class LegalAnswer:
    question: str
    answer: str
    citations: tuple[AnswerCitation, ...]


def citation_from_chunk(chunk: RetrievedChunk) -> AnswerCitation | None:
    """Create a citation only when the chunk has a named source document."""
    if not chunk.source:
        return None
    excerpt = " ".join(chunk.text.split())
    if len(excerpt) > 280:
        excerpt = excerpt[:277].rstrip() + "..."
    return AnswerCitation(
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        section=chunk.section,
        citation=chunk.citation,
        score=chunk.score,
        excerpt=excerpt,
    )


def build_legal_answer(
    generated: GeneratedAnswer,
    chunks: list[RetrievedChunk],
) -> LegalAnswer:
    """Attach citations only for chunks actually supplied to generation."""
    used_ids = set(generated.context_chunk_ids)
    citations = tuple(
        citation
        for chunk in chunks
        if chunk.chunk_id in used_ids
        for citation in [citation_from_chunk(chunk)]
        if citation is not None
    )
    return LegalAnswer(
        question=generated.question,
        answer=generated.answer,
        citations=citations,
    )


def answer_from_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    generator: AnswerGenerator | None = None,
) -> LegalAnswer:
    """Generate a grounded answer and attach its verifiable source citations."""
    generator = generator or AnswerGenerator()
    generated = generator.generate(question, chunks)
    return build_legal_answer(generated, chunks)
