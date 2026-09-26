"""Tests for automatic jurisdiction-aware QA retrieval (issue #22)."""
from legal_core.generation import AnswerGenerator
from legal_core.llm import LLMProvider
from legal_core.retrieval import RetrievedChunk
from legal_core.service import QuestionAnsweringService


def chunk(chunk_id, score, jurisdiction):
    return RetrievedChunk(
        rank=1,
        chunk_id=chunk_id,
        score=score,
        text=f"{jurisdiction} source text.",
        source=f"{chunk_id}.txt",
        section="Section 1",
        chunk_index=0,
        metadata={"jurisdiction": jurisdiction},
    )


class Provider(LLMProvider):
    def generate(self, *, question: str, context: str) -> str:
        return "Grounded answer."


class FilterAwareRetriever:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def retrieve(self, query, top_k=None, *, filters=None):
        jurisdiction = filters.jurisdiction if filters else None
        self.calls.append(jurisdiction)
        return list(self.mapping.get(jurisdiction, []))


class LegacyRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        return [chunk("legacy", 0.9, "Unknown")]


def make_service(retriever):
    return QuestionAnsweringService(
        retriever,
        generator=AnswerGenerator(Provider()),
        min_score=0.2,
    )


def test_provincial_query_combines_province_and_federal_sources():
    retriever = FilterAwareRetriever(
        {
            "Punjab": [chunk("punjab", 0.95, "Punjab")],
            "Federal": [chunk("federal", 0.90, "Federal")],
        }
    )
    result = make_service(retriever).ask("Punjab section 5")
    assert result.status == "answered"
    assert retriever.calls == ["Punjab", "Federal"]
    cited = [c.chunk_id for c in result.answer.citations]
    assert cited == ["punjab", "federal"]


def test_federal_query_does_not_query_province_fallback():
    retriever = FilterAwareRetriever(
        {"Federal": [chunk("federal", 0.9, "Federal")]}
    )
    result = make_service(retriever).ask("Federal Constitution of Pakistan")
    assert result.status == "answered"
    assert retriever.calls == ["Federal"]


def test_empty_jurisdiction_results_fall_back_to_unfiltered_search():
    retriever = FilterAwareRetriever(
        {None: [chunk("broad", 0.85, "Unknown")]}
    )
    result = make_service(retriever).ask("Sindh section 12")
    assert result.status == "answered"
    assert retriever.calls == ["Sindh", "Federal", None]
    assert result.answer.citations[0].chunk_id == "broad"


def test_ambiguous_multi_jurisdiction_query_uses_unfiltered_search():
    retriever = FilterAwareRetriever(
        {None: [chunk("broad", 0.9, "Unknown")]}
    )
    result = make_service(retriever).ask("Compare Punjab and Sindh law")
    assert result.status == "answered"
    assert retriever.calls == [None]


def test_legacy_retriever_without_filters_remains_supported():
    retriever = LegacyRetriever()
    result = make_service(retriever).ask("Punjab section 5")
    assert result.status == "answered"
    assert retriever.calls == 1
