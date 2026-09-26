"""Tests for legal authority-aware reranking (issue #23)."""
from legal_core.authority import (
    assess_authority,
    combine_relevance_and_authority,
)
from legal_core.embeddings import HashingEmbedder
from legal_core.hybrid_retrieval import HybridRetriever
from legal_core.retrieval import Retriever
from legal_core.vector_store import VectorStore


def test_authority_hierarchy_for_known_sources():
    assert assess_authority({"document_type": "statute"}).score == 1.0
    assert assess_authority({"court": "Supreme Court", "document_type": "judgment"}).score == 1.0
    assert assess_authority({"court": "Federal Shariat Court", "document_type": "judgment"}).score == 0.95
    assert assess_authority({"court": "Sindh High Court", "document_type": "judgment"}).score == 0.90
    assert assess_authority({"document_type": "commentary"}).score == 0.55


def test_unknown_authority_has_neutral_midrange_score():
    assessment = assess_authority({})
    assert 0.0 < assessment.score < 1.0
    assert "unknown" in assessment.reason.lower()


def test_authority_signal_does_not_override_large_relevance_gap():
    high_relevance_secondary = combine_relevance_and_authority(
        0.95,
        assess_authority({"document_type": "commentary"}),
    )
    low_relevance_statute = combine_relevance_and_authority(
        0.60,
        assess_authority({"document_type": "statute"}),
    )
    assert high_relevance_secondary > low_relevance_statute


def test_authority_can_break_close_relevance_tie(tmp_path):
    embedder = HashingEmbedder(dimension=64)
    store = VectorStore(tmp_path / "vectors")

    text = "constitutional jurisdiction fundamental rights"
    [embedding] = embedder.embed([text])

    store.upsert(
        "commentary",
        embedding,
        text,
        {
            "source": "commentary.txt",
            "section": "Paragraph 1",
            "chunk_index": 0,
            "document_type": "commentary",
        },
    )
    store.upsert(
        "supreme",
        embedding,
        text,
        {
            "source": "supreme.txt",
            "section": "Paragraph 1",
            "chunk_index": 0,
            "document_type": "judgment",
            "court": "Supreme Court",
        },
    )

    retriever = HybridRetriever(
        Retriever(embedder, store, top_k=2),
        top_k=2,
        authority_weight=0.10,
    )
    results = retriever.retrieve(text)

    assert results[0].chunk_id == "supreme"
    assert results[0].metadata["authority_score"] == 1.0
    assert "Supreme Court" in results[0].metadata["authority_reason"]
    assert all(0.0 <= row.score <= 1.0 for row in results)


def test_zero_authority_weight_preserves_relevance_only_order(tmp_path):
    embedder = HashingEmbedder(dimension=64)
    store = VectorStore(tmp_path / "vectors2")

    [a] = embedder.embed(["contract agreement"])
    [b] = embedder.embed(["contract"])

    store.upsert(
        "a",
        a,
        "contract agreement",
        {"source": "a.txt", "document_type": "commentary"},
    )
    store.upsert(
        "b",
        b,
        "contract",
        {"source": "b.txt", "document_type": "statute"},
    )

    retriever = HybridRetriever(
        Retriever(embedder, store, top_k=2),
        top_k=2,
        authority_weight=0.0,
    )
    results = retriever.retrieve("contract agreement")
    assert results[0].chunk_id == "a"
