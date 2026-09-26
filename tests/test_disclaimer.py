"""Tests that the legal-information disclaimer cannot be omitted (issue #11)."""
from __future__ import annotations

from legal_core.corpus_search import CorpusSearchService
from legal_core.generation import AnswerGenerator
from legal_core.llm import LLMProvider
from legal_core.retrieval import RetrievedChunk
from legal_core.safeguards import LEGAL_INFORMATION_DISCLAIMER
from legal_core.service import QuestionAnsweringService


class StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, top_k=None):
        return list(self.chunks)


class Provider(LLMProvider):
    def generate(self, *, question: str, context: str) -> str:
        return "Grounded answer."


def chunk(score=0.9):
    return RetrievedChunk(
        rank=1,
        chunk_id="c1",
        score=score,
        text="Source text.",
        source="sample.txt",
        section="Section 1",
        chunk_index=0,
    )


def test_answered_qa_response_contains_disclaimer():
    result = QuestionAnsweringService(
        StubRetriever([chunk()]),
        generator=AnswerGenerator(Provider()),
        min_score=0.2,
    ).ask("Question?")

    assert result.disclaimer == LEGAL_INFORMATION_DISCLAIMER
    assert result.answer is not None
    assert result.answer.disclaimer == LEGAL_INFORMATION_DISCLAIMER


def test_no_source_qa_response_still_contains_disclaimer():
    result = QuestionAnsweringService(
        StubRetriever([]),
        generator=AnswerGenerator(Provider()),
        min_score=0.2,
    ).ask("Question?")
    assert result.answer is None
    assert result.disclaimer == LEGAL_INFORMATION_DISCLAIMER


def test_corpus_search_response_contains_disclaimer():
    response = CorpusSearchService(StubRetriever([chunk()])).search("source")
    assert response.disclaimer == LEGAL_INFORMATION_DISCLAIMER
    assert "not legal advice" in response.disclaimer.lower()
