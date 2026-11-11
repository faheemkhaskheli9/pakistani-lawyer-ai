"""Tests for issue #2's ingest-store -> embedder -> vector-store wiring."""

from __future__ import annotations

from legal_core.chunk_store import ChunkRecord, ChunkStore
from legal_core.embedding_pipeline import embed_new_chunks
from legal_core.embeddings import HashingEmbedder
from legal_core.vector_store import VectorStore


def _record(id_: str, text: str) -> ChunkRecord:
    return ChunkRecord(id=id_, source="statute.txt", section="s1", chunk_index=0, text=text)


def test_embeds_every_chunk_on_first_run(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.json")
    chunk_store.upsert([_record("c1", "contract law basics"), _record("c2", "criminal procedure")])
    vector_store = VectorStore(tmp_path / "vectors")

    n = embed_new_chunks(chunk_store, HashingEmbedder(dimension=32), vector_store)

    assert n == 2
    assert len(vector_store) == 2
    assert "c1" in vector_store and "c2" in vector_store


def test_rerunning_with_no_new_chunks_is_a_no_op(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.json")
    chunk_store.upsert([_record("c1", "contract law basics")])
    vector_store = VectorStore(tmp_path / "vectors")
    embed_new_chunks(chunk_store, HashingEmbedder(dimension=32), vector_store)

    class ExplodingEmbedder(HashingEmbedder):
        def embed(self, texts):
            raise AssertionError("embedder should not be called for already-embedded chunks")

    n = embed_new_chunks(chunk_store, ExplodingEmbedder(dimension=32), vector_store)

    assert n == 0


def test_only_new_chunks_are_embedded_on_a_subsequent_run(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.json")
    chunk_store.upsert([_record("c1", "contract law basics")])
    vector_store = VectorStore(tmp_path / "vectors")
    embed_new_chunks(chunk_store, HashingEmbedder(dimension=32), vector_store)

    chunk_store.upsert([_record("c2", "criminal procedure")])
    n = embed_new_chunks(chunk_store, HashingEmbedder(dimension=32), vector_store)

    assert n == 1
    assert len(vector_store) == 2


def test_vector_store_metadata_carries_source_and_section(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.json")
    chunk_store.upsert([_record("c1", "contract law basics")])
    vector_store = VectorStore(tmp_path / "vectors")

    embed_new_chunks(chunk_store, HashingEmbedder(dimension=32), vector_store)

    results = vector_store.query(HashingEmbedder(dimension=32).embed(["contract law basics"])[0], top_k=1)
    assert results[0].metadata["source"] == "statute.txt"
    assert results[0].metadata["section"] == "s1"

def test_deleted_or_amended_source_is_removed_from_vector_index(tmp_path):
    from legal_core.ingest import run_ingestion

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "act.txt"
    doc.write_text("Section 1. Old rule.", encoding="utf-8")
    index = tmp_path / "index"
    run_ingestion(corpus, index, embedding_provider="hashing")
    original = VectorStore(index / "vectors")
    assert len(original) > 0

    doc.write_text("Section 1. New rule.", encoding="utf-8")
    run_ingestion(corpus, index, embedding_provider="hashing")
    updated = VectorStore(index / "vectors")
    assert len(updated) > 0
    assert all("Old rule" not in entry.document for entry in updated.entries())
    assert any("New rule" in entry.document for entry in updated.entries())

    doc.unlink()
    run_ingestion(corpus, index, embedding_provider="hashing")
    assert len(VectorStore(index / "vectors")) == 0


def test_metadata_version_change_replaces_indexed_source(tmp_path):
    import json

    from legal_core.ingest import run_ingestion

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "act.txt").write_text("Section 1. Valid text.", encoding="utf-8")
    manifest = corpus / "legal_metadata.json"
    index = tmp_path / "index"
    manifest.write_text(json.dumps({"act.txt": {"effective_date": "2020-01-01"}}), encoding="utf-8")
    run_ingestion(corpus, index)
    manifest.write_text(json.dumps({"act.txt": {"effective_date": "2025-01-01"}}), encoding="utf-8")
    run_ingestion(corpus, index)
    entries = VectorStore(index / "vectors").entries()
    assert entries
    assert all(row.metadata["effective_date"] == "2025-01-01" for row in entries)
