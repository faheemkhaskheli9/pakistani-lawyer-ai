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
