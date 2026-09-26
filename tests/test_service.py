"""Tests for no-relevant-source gating (issue #9)."""
from __future__ import annotations

import pytest

from legal_core.generation import AnswerGenerator
from legal_core.llm import LLMProvider
from legal_core.retrieval import RetrievedChunk
from legal_core.service import (
    DEFAULT_MIN_SCORE,
    NO_RELEVANT_SOURCE_MESSAGE,
    QuestionAnsweringService,
    resolve_min_score,
)


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return list(self.results)


class RecordingProvider(LLMProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, *, question: str, context: str) -> str:
        self.calls += 1
        return "Grounded answer."


def chunk(score: float, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        chunk_id=chunk_id,
        score=score,
        text="Relevant legal source text.",
        source="sample.txt",
        section="Section 1",
        chunk_index=0,
    )


def test_out_of_corpus_query_returns_fallback_without_generation():
    provider = RecordingProvider()
    service = QuestionAnsweringService(
        StubRetriever([chunk(0.05)]),
        generator=AnswerGenerator(provider),
        min_score=0.40,
    )

    result = service.ask("An unrelated question outside the corpus")

    assert result.status == "no_relevant_source"
    assert result.answer is None
    assert result.message == NO_RELEVANT_SOURCE_MESSAGE
    assert provider.calls == 0


def test_empty_retrieval_returns_same_safe_fallback():
    provider = RecordingProvider()
    result = QuestionAnsweringService(
        StubRetriever([]),
        generator=AnswerGenerator(provider),
        min_score=0.20,
    ).ask("Question?")

    assert result.found_relevant_source is False
    assert result.answer is None
    assert provider.calls == 0


def test_only_chunks_meeting_threshold_are_sent_to_generation():
    provider = RecordingProvider()
    service = QuestionAnsweringService(
        StubRetriever([chunk(0.91, "high"), chunk(0.10, "low")]),
        generator=AnswerGenerator(provider),
        min_score=0.50,
    )

    result = service.ask("Question?")

    assert result.status == "answered"
    assert result.answer is not None
    assert [c.chunk_id for c in result.answer.citations] == ["high"]
    assert provider.calls == 1


def test_threshold_defaults_and_environment_override():
    assert resolve_min_score(env={}) == DEFAULT_MIN_SCORE
    assert resolve_min_score(env={"RETRIEVAL_MIN_SCORE": "0.55"}) == 0.55
    assert resolve_min_score(0.75, env={"RETRIEVAL_MIN_SCORE": "0.10"}) == 0.75


@pytest.mark.parametrize("value", [-0.01, 1.01, "bad", True])
def test_invalid_explicit_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="between 0 and 1"):
        resolve_min_score(value)


@pytest.mark.parametrize("raw", ["-0.1", "1.1", "not-a-number"])
def test_invalid_environment_threshold_is_rejected(raw):
    with pytest.raises(ValueError, match="RETRIEVAL_MIN_SCORE"):
        resolve_min_score(env={"RETRIEVAL_MIN_SCORE": raw})


def test_blank_question_is_rejected_before_retrieval():
    retriever = StubRetriever([chunk(0.9)])
    service = QuestionAnsweringService(retriever, min_score=0.2)
    with pytest.raises(ValueError, match="question"):
        service.ask("   ")
    assert retriever.queries == []
