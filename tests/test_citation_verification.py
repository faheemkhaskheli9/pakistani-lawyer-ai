"""Tests for citation integrity verification (issue #17)."""
from __future__ import annotations

from dataclasses import replace

from legal_core.citation_verification import verify_citations
from legal_core.generation import AnswerGenerator
from legal_core.llm import LLMProvider
from legal_core.qa import AnswerCitation, LegalAnswer
from legal_core.retrieval import RetrievedChunk
from legal_core.service import QuestionAnsweringService


def chunk(chunk_id="c1", source="law.txt", section="Section 5", score=0.9):
    return RetrievedChunk(
        rank=1,
        chunk_id=chunk_id,
        score=score,
        text="Grounded legal text.",
        source=source,
        section=section,
        chunk_index=0,
    )


def citation(chunk_id="c1", source="law.txt", section="Section 5"):
    return AnswerCitation(
        chunk_id=chunk_id,
        source=source,
        section=section,
        citation=f"{source}, {section}",
        score=0.9,
        excerpt="Grounded legal text.",
    )


def answer(*citations):
    return LegalAnswer(question="Q?", answer="A.", citations=tuple(citations))


def test_valid_citation_is_verified():
    report = verify_citations(answer(citation()), [chunk()])
    assert report.verified is True
    assert report.verified_citations == 1
    assert report.checks[0].verified is True


def test_missing_grounding_chunk_is_rejected():
    report = verify_citations(answer(citation(chunk_id="fabricated")), [chunk()])
    assert report.verified is False
    assert "does not reference" in report.checks[0].reason


def test_tampered_source_or_section_is_rejected():
    bad = citation(source="other.txt")
    report = verify_citations(answer(bad), [chunk()])
    assert not report.verified
    assert "source" in report.checks[0].reason


def test_duplicate_citation_is_not_fully_verified():
    c = citation()
    report = verify_citations(answer(c, c), [chunk()])
    assert report.verified is False
    assert any("duplicate" in check.reason for check in report.checks)


def test_answer_without_citations_is_not_verified():
    report = verify_citations(answer(), [chunk()])
    assert report.verified is False
    assert report.total_citations == 0


class Provider(LLMProvider):
    def generate(self, *, question: str, context: str) -> str:
        return "Grounded answer."


class Retriever:
    def retrieve(self, query):
        return [chunk()]


def test_qa_service_exposes_verified_citation_report():
    result = QuestionAnsweringService(
        Retriever(),
        generator=AnswerGenerator(Provider()),
        min_score=0.2,
    ).ask("Question?")

    assert result.status == "answered"
    assert result.answer is not None
    assert result.citation_verification is not None
    assert result.citation_verification.verified is True
