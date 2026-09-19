"""Tests for the `python -m legal_core.ingest` CLI (issue #4, Phase 1).

The default embedding provider (`HashingEmbedder`, see `legal_core.embeddings`)
is a deterministic, offline, CPU-only bag-of-words embedder -- there is no
network call or model download to mock here, so these tests exercise the
real `main()` entry point end to end against a temporary corpus/index dir.
"""
from __future__ import annotations

from legal_core.chunk_store import ChunkStore
from legal_core.ingest import main
from legal_core.seed_corpus import SEED_CORPUS_DIR
from legal_core.vector_store import VectorStore


def _make_corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "act.txt").write_text(
        "Section 1. Short title. " * 20 + "Section 2. Definitions. " * 20,
        encoding="utf-8",
    )
    return corpus_dir


def test_cli_runs_ingestion_and_embedding_end_to_end(tmp_path):
    corpus_dir = _make_corpus(tmp_path)
    index_dir = tmp_path / "index"

    exit_code = main(["--corpus", str(corpus_dir), "--index-dir", str(index_dir)])

    assert exit_code == 0
    chunk_store = ChunkStore(index_dir / "chunks.json")
    vector_store = VectorStore(index_dir / "vectors")
    assert len(chunk_store) > 0
    assert len(vector_store) == len(chunk_store)
    # Every ingested chunk was actually embedded into the vector store.
    assert all(record.id in vector_store for record in chunk_store.all())


def test_cli_runs_against_the_seed_corpus(tmp_path):
    index_dir = tmp_path / "index"

    exit_code = main(["--corpus", str(SEED_CORPUS_DIR), "--index-dir", str(index_dir)])

    assert exit_code == 0
    chunk_store = ChunkStore(index_dir / "chunks.json")
    vector_store = VectorStore(index_dir / "vectors")
    assert len(chunk_store) > 0
    assert len(vector_store) == len(chunk_store)


def test_cli_rerun_does_not_duplicate_the_index(tmp_path):
    corpus_dir = _make_corpus(tmp_path)
    index_dir = tmp_path / "index"

    first_exit = main(["--corpus", str(corpus_dir), "--index-dir", str(index_dir)])
    chunk_store_after_first = ChunkStore(index_dir / "chunks.json")
    vector_store_after_first = VectorStore(index_dir / "vectors")
    count_after_first = len(chunk_store_after_first)
    vector_count_after_first = len(vector_store_after_first)

    second_exit = main(["--corpus", str(corpus_dir), "--index-dir", str(index_dir)])
    chunk_store_after_second = ChunkStore(index_dir / "chunks.json")
    vector_store_after_second = VectorStore(index_dir / "vectors")

    assert first_exit == 0
    assert second_exit == 0
    assert len(chunk_store_after_second) == count_after_first
    assert len(vector_store_after_second) == vector_count_after_first


def test_cli_exits_non_zero_on_missing_corpus_dir(tmp_path, capsys):
    index_dir = tmp_path / "index"

    exit_code = main(["--corpus", str(tmp_path / "does-not-exist"), "--index-dir", str(index_dir)])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Ingestion failed" in captured.err
    # A failed run must not leave a partially-created index behind for a
    # caller to mistake for a real one.
    assert not (index_dir / "chunks.json").exists()


def test_cli_exits_non_zero_on_unknown_embedding_provider(tmp_path):
    corpus_dir = _make_corpus(tmp_path)
    index_dir = tmp_path / "index"

    exit_code = main(
        [
            "--corpus", str(corpus_dir),
            "--index-dir", str(index_dir),
            "--embedding-provider", "not-a-real-provider",
        ]
    )

    assert exit_code != 0
