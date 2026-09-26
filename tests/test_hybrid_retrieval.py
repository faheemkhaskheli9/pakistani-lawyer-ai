"""Tests for hybrid BM25 + vector retrieval (issue #16)."""
from __future__ import annotations

from legal_core.embeddings import EmbeddingProvider
from legal_core.hybrid_retrieval import BM25Index, HybridRetriever, tokenize
from legal_core.retrieval import Retriever
from legal_core.vector_store import VectorStore


class AxisEmbedder(EmbeddingProvider):
    KEYWORDS = ("contract", "bail", "evidence")

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts):
        return [
            [float(text.lower().count(word)) for word in self.KEYWORDS]
            for text in texts
        ]


def make_store(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    embedder = AxisEmbedder()
    docs = [
        ("contract", "Contract formation and free consent.", "civil.txt", "Section 3"),
        ("bail", "Bail may be granted by a competent court.", "criminal.txt", "Section 8"),
        ("citation", "Article 199 constitutional jurisdiction PLD 2024 Demo 42.", "case.txt", "Article 199"),
    ]
    for chunk_id, text, source, section in docs:
        [embedding] = embedder.embed([text])
        store.upsert(
            chunk_id,
            embedding,
            text,
            {"source": source, "section": section, "chunk_index": 0},
        )
    return store, embedder


def test_tokenizer_preserves_legal_numbers():
    assert tokenize("Article 199, PLD-2024") == ["article", "199", "pld", "2024"]


def test_bm25_finds_exact_citation_terms(tmp_path):
    store, _ = make_store(tmp_path)
    matches = BM25Index(store.entries()).search("Article 199 PLD 2024", top_k=3)
    assert matches[0].id == "citation"
    assert matches[0].score == 1.0


def test_hybrid_can_retrieve_lexical_match_when_embedding_query_has_no_signal(tmp_path):
    store, embedder = make_store(tmp_path)
    hybrid = HybridRetriever(Retriever(embedder, store, top_k=2), top_k=2)

    results = hybrid.retrieve("Article 199 PLD 2024")

    assert results
    assert results[0].chunk_id == "citation"
    assert 0.0 <= results[0].score <= 1.0


def test_hybrid_combines_vector_and_lexical_signals(tmp_path):
    store, embedder = make_store(tmp_path)
    hybrid = HybridRetriever(Retriever(embedder, store, top_k=3), vector_weight=0.6, top_k=3)

    results = hybrid.retrieve("contract free consent")

    assert results[0].chunk_id == "contract"
    assert results[0].source == "civil.txt"
    assert results[0].section == "Section 3"
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_hybrid_respects_top_k(tmp_path):
    store, embedder = make_store(tmp_path)
    hybrid = HybridRetriever(Retriever(embedder, store, top_k=3), top_k=3)
    assert len(hybrid.retrieve("contract bail evidence", top_k=1)) == 1


def test_hybrid_empty_store_returns_empty(tmp_path):
    store = VectorStore(tmp_path / "empty")
    hybrid = HybridRetriever(Retriever(AxisEmbedder(), store, top_k=3), top_k=3)
    assert hybrid.retrieve("Article 199") == []
