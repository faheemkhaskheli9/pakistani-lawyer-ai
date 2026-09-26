"""Tests for citation-carrying legal answers (issue #8)."""
from __future__ import annotations

from legal_core.generation import AnswerGenerator, GeneratedAnswer
from legal_core.llm import LLMProvider
from legal_core.qa import answer_from_chunks, build_legal_answer
from legal_core.retrieval import RetrievedChunk


class StaticProvider(LLMProvider):
    def generate(self, *, question: str, context: str) -> str:
        return "The retrieved source addresses the question."


def chunk(
    chunk_id: str,
    *,
    source: str | None = "sample_civil_code.txt",
    section: str | None = "Section 3",
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        score=0.91,
        text="Consent is not free when caused by coercion or fraud.",
        source=source,
        section=section,
        chunk_index=rank - 1,
    )


def test_answer_includes_specific_source_and_section_citations():
    response = answer_from_chunks(
        "What affects free consent?",
        [chunk("civil-3")],
        generator=AnswerGenerator(StaticProvider()),
    )

    assert response.answer
    assert len(response.citations) == 1
    citation = response.citations[0]
    assert citation.source == "sample_civil_code.txt"
    assert citation.section == "Section 3"
    assert citation.citation == "sample_civil_code.txt, Section 3"
    assert citation.chunk_id == "civil-3"
    assert "Consent is not free" in citation.excerpt


def test_all_grounding_chunks_are_exposed_as_citations():
    response = answer_from_chunks(
        "Question?",
        [
            chunk("one", section="Section 3", rank=1),
            chunk("two", section="Section 5", rank=2),
        ],
        generator=AnswerGenerator(StaticProvider()),
    )
    assert [c.chunk_id for c in response.citations] == ["one", "two"]


def test_chunk_not_used_for_generation_is_not_presented_as_cited():
    used = chunk("used")
    unrelated = chunk("not-used", section="Section 99", rank=2)
    generated = GeneratedAnswer(
        question="Question?",
        answer="Answer.",
        context_chunk_ids=("used",),
    )

    response = build_legal_answer(generated, [used, unrelated])

    assert [c.chunk_id for c in response.citations] == ["used"]


def test_chunk_without_named_source_is_not_presented_as_verifiable_citation():
    generated = GeneratedAnswer(
        question="Question?",
        answer="Answer.",
        context_chunk_ids=("anonymous",),
    )
    response = build_legal_answer(
        generated,
        [chunk("anonymous", source=None, section=None)],
    )
    assert response.citations == ()
