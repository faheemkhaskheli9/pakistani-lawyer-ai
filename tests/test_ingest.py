from __future__ import annotations

import pytest

from legal_core.chunk_store import ChunkStore
from legal_core.ingest import IngestionPipeline, iter_source_files
from legal_core.parsing import DocumentParseError


def make_pipeline(tmp_path):
    store = ChunkStore(tmp_path / "chunks.json")
    return IngestionPipeline(store, chunk_size=200, chunk_overlap=20), store


def test_ingest_file_stores_chunks_with_source_and_section(tmp_path):
    pipeline, store = make_pipeline(tmp_path)
    doc = tmp_path / "act.txt"
    doc.write_text("Section 1. Short title. " * 20, encoding="utf-8")

    count = pipeline.ingest_file(doc)

    assert count > 0
    assert len(store) == count
    for record in store.all():
        assert record.source == str(doc)
        assert record.section  # e.g. "Section 1"
        assert record.text


def test_reingesting_unchanged_corpus_does_not_duplicate_chunks(tmp_path):
    pipeline, store = make_pipeline(tmp_path)
    doc = tmp_path / "act.txt"
    doc.write_text("Section 1. Short title. " * 20, encoding="utf-8")

    first_count = pipeline.ingest_file(doc)
    second_count = pipeline.ingest_file(doc)

    assert second_count == first_count
    assert len(store) == first_count  # upserted, not appended


def test_editing_a_chunk_creates_a_new_id_not_a_silent_overwrite(tmp_path):
    pipeline, store = make_pipeline(tmp_path)
    doc = tmp_path / "act.txt"
    doc.write_text("Section 1. Original text.", encoding="utf-8")
    pipeline.ingest_file(doc)
    original_ids = store.get_ids()

    doc.write_text("Section 1. Edited text.", encoding="utf-8")
    pipeline.ingest_file(doc)

    assert store.get_ids() != original_ids
    texts = {r.text for r in store.all()}
    assert any("Edited text" in t for t in texts)


def test_ingest_file_raises_for_malformed_document(tmp_path):
    pipeline, _store = make_pipeline(tmp_path)
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a real pdf")
    with pytest.raises(DocumentParseError):
        pipeline.ingest_file(bad)


def test_ingest_directory_skips_malformed_document_and_continues(tmp_path):
    pipeline, store = make_pipeline(tmp_path)
    good = tmp_path / "good.txt"
    good.write_text("Section 1. This one parses fine. " * 10, encoding="utf-8")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a real pdf")

    total = pipeline.ingest_directory(tmp_path)

    assert total > 0
    assert len(store) == total  # only the good file's chunks landed


def test_ingest_directory_missing_source_dir_raises():
    from legal_core.chunk_store import ChunkStore as _CS

    pipeline = IngestionPipeline(_CS("/nonexistent/does-not-exist/chunks.json"))
    with pytest.raises(FileNotFoundError):
        pipeline.ingest_directory("/nonexistent/does-not-exist-dir")


def test_iter_source_files_only_returns_supported_extensions(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.docx").write_text("x", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4")

    files = iter_source_files(tmp_path)

    names = {p.name for p in files}
    assert names == {"a.txt", "c.pdf"}
