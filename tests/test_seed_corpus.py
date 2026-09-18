"""Tests for the checked-in seed corpus (issue #3, Phase 1).

Covers the acceptance criteria directly: the seed corpus loads and chunks
through the real `IngestionPipeline`, each seed document has a recorded
source/license basis, and the resulting chunks embed end-to-end into a
vector store through the existing `legal_core` pipeline -- no seed-only
code path.
"""
from __future__ import annotations

import pytest

from legal_core.chunk_store import ChunkStore
from legal_core.embedding_pipeline import embed_new_chunks
from legal_core.embeddings import HashingEmbedder
from legal_core.seed_corpus import SEED_CORPUS_DIR, load_seed_corpus, load_sources_manifest
from legal_core.vector_store import VectorStore


def test_seed_corpus_directory_has_at_least_a_few_documents():
    seed_files = sorted(SEED_CORPUS_DIR.glob("*.txt"))
    assert len(seed_files) >= 3


def test_seed_corpus_is_small_enough_to_commit_directly():
    # Acceptance criterion: no large binary downloads required for tests --
    # assert the whole checked-in seed corpus is well under 1 MB.
    total_bytes = sum(p.stat().st_size for p in SEED_CORPUS_DIR.glob("*.txt"))
    assert total_bytes < 1_000_000


def test_every_seed_document_has_a_recorded_source_and_license_basis():
    manifest = load_sources_manifest()
    seed_files = {p.name for p in SEED_CORPUS_DIR.glob("*.txt")}

    assert seed_files  # sanity: there is actually a corpus to check
    assert seed_files <= manifest.keys()
    for filename in seed_files:
        entry = manifest[filename]
        assert entry["title"]
        assert entry["basis"]  # the public-domain/open-license basis, non-empty


def test_manifest_missing_entry_raises(tmp_path):
    (tmp_path / "undocumented.txt").write_text("Section 1. Some text.", encoding="utf-8")
    (tmp_path / "sources.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        load_sources_manifest(tmp_path)


def test_manifest_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sources_manifest(tmp_path)


def test_load_seed_corpus_ingests_chunks_with_source_and_section(tmp_path):
    store = ChunkStore(tmp_path / "chunks.json")

    count = load_seed_corpus(store)

    assert count > 0
    assert len(store) == count
    sources = {record.source for record in store.all()}
    # Every seed .txt file actually produced chunks.
    assert len(sources) == len(list(SEED_CORPUS_DIR.glob("*.txt")))
    for record in store.all():
        assert record.text.strip()
        assert record.section  # e.g. "Section 1" or a positional fallback


def test_load_seed_corpus_is_idempotent(tmp_path):
    store = ChunkStore(tmp_path / "chunks.json")

    first_count = load_seed_corpus(store)
    second_count = load_seed_corpus(store)

    assert second_count == first_count
    assert len(store) == first_count  # upserted, not duplicated


def test_seed_corpus_embeds_end_to_end_through_existing_pipeline(tmp_path):
    chunk_store = ChunkStore(tmp_path / "chunks.json")
    vector_store = VectorStore(tmp_path / "vectors")

    chunk_count = load_seed_corpus(chunk_store)
    embedded_count = embed_new_chunks(chunk_store, HashingEmbedder(dimension=64), vector_store)

    assert embedded_count == chunk_count
    assert len(vector_store) == chunk_count

    # A query about contracts should surface the civil-code chunk with a
    # citation back to its source document and section.
    query_vector = HashingEmbedder(dimension=64).embed(["capacity to contract"])[0]
    results = vector_store.query(query_vector, top_k=1)
    assert results
    assert "sample_civil_code" in results[0].metadata["source"]
    assert results[0].metadata["section"]
