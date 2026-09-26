"""Tests for no-LLM corpus browse/search (issue #10)."""
from __future__ import annotations

from legal_core.corpus_search import CorpusSearchService
from legal_core.retrieval import RetrievedChunk


class StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, query, top_k=None):
        self.calls.append((query, top_k))
        return list(self.chunks)


def chunk(source="sample_evidence_act.txt"):
    return RetrievedChunk(
        rank=1,
        chunk_id="evidence-1",
        score=0.88,
        text="Section 4. Burden of Proof. The burden rests on the party who asserts the fact.",
        source=source,
        section="Section 4",
        chunk_index=0,
    )


def test_search_returns_direct_source_results_without_generation_dependency():
    retriever = StubRetriever([chunk()])
    service = CorpusSearchService(retriever)

    response = service.search("burden of proof", top_k=3)

    assert retriever.calls == [("burden of proof", 3)]
    assert len(response.results) == 1
    result = response.results[0]
    assert result.source == "sample_evidence_act.txt"
    assert result.section == "Section 4"
    assert result.citation == "sample_evidence_act.txt, Section 4"
    assert result.source_path == "data/corpus/sample_evidence_act.txt"
    assert "burden rests" in result.excerpt


def test_search_drops_chunks_without_a_browsable_source():
    service = CorpusSearchService(StubRetriever([chunk(source=None)]))
    assert service.search("anything").results == ()


def test_source_path_uses_filename_not_untrusted_parent_path():
    result = CorpusSearchService(
        StubRetriever([chunk(source="../../secret.txt")])
    ).search("anything").results[0]
    assert result.source == "secret.txt"
    assert result.source_path == "data/corpus/secret.txt"


def test_blank_search_query_is_rejected():
    service = CorpusSearchService(StubRetriever([]))
    try:
        service.search("   ")
    except ValueError as exc:
        assert "query" in str(exc)
    else:
        raise AssertionError("blank query should fail")
