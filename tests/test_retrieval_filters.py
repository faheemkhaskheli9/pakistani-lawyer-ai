"""Tests for metadata-aware retrieval filters (issue #20)."""
from __future__ import annotations

import pytest

from legal_core.embeddings import HashingEmbedder
from legal_core.filters import RetrievalFilters, matches_filters
from legal_core.hybrid_retrieval import HybridRetriever
from legal_core.retrieval import Retriever
from legal_core.vector_store import VectorStore


def build(tmp_path):
    embedder = HashingEmbedder(dimension=64)
    store = VectorStore(tmp_path / "vectors")
    records = [
        (
            "sc",
            "Constitutional jurisdiction and fundamental rights.",
            {
                "source": "sc.txt",
                "section": "Paragraph 10",
                "chunk_index": 0,
                "jurisdiction": "Federal",
                "court": "Supreme Court",
                "document_type": "judgment",
                "case_citation": "PLD 2026 SC 1",
                "decision_date": "2026-01-10",
            },
        ),
        (
            "sindh",
            "Constitutional petition before the High Court.",
            {
                "source": "shc.txt",
                "section": "Paragraph 5",
                "chunk_index": 0,
                "jurisdiction": "Sindh",
                "court": "Sindh High Court",
                "document_type": "judgment",
                "case_citation": "2025 CLC 10",
                "decision_date": "2025-06-15",
            },
        ),
        (
            "act",
            "A federal statute concerning contracts.",
            {
                "source": "act.txt",
                "section": "Section 1",
                "chunk_index": 0,
                "jurisdiction": "Federal",
                "document_type": "statute",
                "effective_date": "2024-01-01",
            },
        ),
    ]
    for chunk_id, text, metadata in records:
        [embedding] = embedder.embed([text])
        store.upsert(chunk_id, embedding, text, metadata)
    return HybridRetriever(Retriever(embedder, store, top_k=5), top_k=5)


def test_exact_metadata_filters_are_case_insensitive():
    metadata = {"jurisdiction": "Sindh", "court": "Sindh High Court"}
    assert matches_filters(
        metadata,
        RetrievalFilters(jurisdiction="sindh", court="sindh high court"),
    )


def test_hybrid_retrieval_filters_by_jurisdiction_and_document_type(tmp_path):
    retriever = build(tmp_path)
    results = retriever.retrieve(
        "constitutional jurisdiction",
        filters=RetrievalFilters(jurisdiction="Federal", document_type="judgment"),
    )
    assert [row.chunk_id for row in results] == ["sc"]


def test_case_citation_filter_selects_exact_case(tmp_path):
    retriever = build(tmp_path)
    results = retriever.retrieve(
        "constitutional",
        filters=RetrievalFilters(case_citation="2025 CLC 10"),
    )
    assert [row.chunk_id for row in results] == ["sindh"]


def test_date_range_filters_decision_or_effective_date(tmp_path):
    retriever = build(tmp_path)
    results = retriever.retrieve(
        "federal",
        filters=RetrievalFilters(date_from="2024-01-01", date_to="2024-12-31"),
    )
    assert [row.chunk_id for row in results] == ["act"]


def test_nonmatching_filter_returns_empty(tmp_path):
    retriever = build(tmp_path)
    assert retriever.retrieve(
        "constitutional",
        filters=RetrievalFilters(jurisdiction="Punjab"),
    ) == []


def test_invalid_date_range_is_rejected():
    with pytest.raises(ValueError, match="date_from"):
        RetrievalFilters(date_from="01/01/2025")
    with pytest.raises(ValueError, match="on or before"):
        RetrievalFilters(date_from="2026-01-01", date_to="2025-01-01")
