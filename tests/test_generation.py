"""Tests for grounded answer generation (issue #7)."""
from __future__ import annotations

import pytest

from legal_core.generation import AnswerGenerator, GenerationError
from legal_core.llm import LLMProvider, LocalExtractiveProvider, get_llm_provider
from legal_core.retrieval import RetrievedChunk


def chunk(
    rank: int,
    chunk_id: str,
    text: str,
    source: str = "sample.txt",
    section: str = "Section 1",
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        score=0.9,
        text=text,
        source=source,
        section=section,
        chunk_index=rank - 1,
    )


class RecordingProvider(LLMProvider):
    def __init__(self):
        self.calls = []

    def generate(self, *, question: str, context: str) -> str:
        self.calls.append({"question": question, "context": context})
        return "Grounded answer from supplied context."


class FailingProvider(LLMProvider):
    def generate(self, *, question: str, context: str) -> str:
        raise RuntimeError("provider unavailable")


def test_generation_is_conditioned_on_retrieved_chunks_only():
    provider = RecordingProvider()
    generator = AnswerGenerator(provider)
    chunks = [
        chunk(1, "a", "A contract requires free consent.", section="Section 3"),
        chunk(2, "b", "Damages may follow breach.", section="Section 5"),
    ]

    result = generator.generate("What does the corpus say about contracts?", chunks)

    assert result.context_chunk_ids == ("a", "b")
    assert len(provider.calls) == 1
    sent = provider.calls[0]
    assert sent["question"] == "What does the corpus say about contracts?"
    assert "A contract requires free consent." in sent["context"]
    assert "Damages may follow breach." in sent["context"]
    assert "Section 3" in sent["context"]
    assert "Section 5" in sent["context"]


def test_default_provider_is_offline_local_provider():
    assert isinstance(get_llm_provider(), LocalExtractiveProvider)
    answer = AnswerGenerator().generate(
        "What is relevant?",
        [chunk(1, "a", "Only retrieved text may be used.")],
    )
    assert "Only retrieved text may be used." in answer.answer


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_provider("does-not-exist")


def test_generation_failure_is_exposed_as_clear_error():
    generator = AnswerGenerator(FailingProvider())
    with pytest.raises(GenerationError, match="provider unavailable"):
        generator.generate("Question?", [chunk(1, "a", "Grounding")])


@pytest.mark.parametrize("question", ["", "   ", None])
def test_blank_question_is_rejected(question):
    with pytest.raises(ValueError, match="question"):
        AnswerGenerator(RecordingProvider()).generate(
            question,
            [chunk(1, "a", "Grounding")],
        )


def test_generation_without_retrieved_context_is_rejected():
    with pytest.raises(ValueError, match="retrieved chunk"):
        AnswerGenerator(RecordingProvider()).generate("Question?", [])
