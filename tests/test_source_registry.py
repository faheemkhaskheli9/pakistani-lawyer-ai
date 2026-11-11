"""Fail-closed source admission tests. Legal text here is synthetic test data."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.chunk_store import ChunkStore
from legal_core.ingest import run_ingestion
from legal_core.source_registry import SourceRegistryError, validate_registry
from legal_core.vector_store import VectorStore


def fixture_corpus(tmp_path, *, kind="statute"):
    root = tmp_path / "corpus"
    root.mkdir()
    doc = root / "test_act.txt"
    doc.write_text(
        "Section 1. Synthetic test statute created exclusively for unit tests. "
        "No assertion in this document states actual Pakistani law.",
        encoding="utf-8",
    )
    today = date.today().isoformat()
    entry = {
        "title": "Synthetic test statute",
        "document_type": kind,
        "jurisdiction": "Federal",
        "publisher": "National Assembly of Pakistan",
        "source_url": "https://www.na.gov.pk/en/downloads.php",
        "sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
        "legal_status": "current",
        "rights_status": "authorized",
        "rights_basis": "Fixture is created by the test author; not an actual statute.",
        "rights_reference": "internal tests/test_source_registry.py fixture",
        "reviewer": "fixture-test-reviewer",
        "reviewed_on": today,
        "effective_date": "2025-01-01",
        "gazette_url": "https://pakistancode.gov.pk/english/",
        "gazette_checked_on": today,
    }
    if kind == "judgment":
        del entry["effective_date"]
        del entry["gazette_url"]
        del entry["gazette_checked_on"]
        entry["decision_date"] = "2025-01-01"
        entry["court"] = "Supreme Court"
    registry = root / "source_registry.json"

    def save():
        registry.write_text(
            json.dumps({"schema_version": 1, "documents": {doc.name: entry}}),
            encoding="utf-8",
        )

    save()
    return root, doc, entry, save


def test_audited_local_document_can_be_admitted(tmp_path):
    root, doc, entry, save = fixture_corpus(tmp_path)
    records = validate_registry(root)
    assert list(records) == [doc.name]
    assert records[doc.name]["corpus_admitted"] is True
    assert records[doc.name]["source_sha256"] == entry["sha256"]
    assert records[doc.name]["gazette_url"] == entry["gazette_url"]


def test_judgment_registry_requires_court_and_decision_date(tmp_path):
    root, doc, entry, save = fixture_corpus(tmp_path, kind="judgment")
    assert validate_registry(root)[doc.name]["court"] == "Supreme Court"
    del entry["court"]
    save()
    with pytest.raises(SourceRegistryError, match="court"):
        validate_registry(root)


def test_changed_file_fails_before_index_update(tmp_path):
    root, doc, entry, save = fixture_corpus(tmp_path)
    index = tmp_path / "index"
    run_ingestion(root, index, admitted_only=True)
    old_ids = ChunkStore(index / "chunks.json").get_ids()

    doc.write_text("Section 2. Unexpected replacement data.", encoding="utf-8")
    with pytest.raises(SourceRegistryError, match="SHA-256 mismatch"):
        run_ingestion(root, index, admitted_only=True)
    assert ChunkStore(index / "chunks.json").get_ids() == old_ids


def test_approved_update_replaces_old_law_chunks_and_vectors(tmp_path):
    root, doc, entry, save = fixture_corpus(tmp_path)
    index = tmp_path / "index"
    run_ingestion(root, index, admitted_only=True)

    doc.write_text(
        "Section 1. Entirely revised synthetic fixture about a different rule.",
        encoding="utf-8",
    )
    entry["sha256"] = hashlib.sha256(doc.read_bytes()).hexdigest()
    save()
    run_ingestion(root, index, admitted_only=True)
    assert all("created exclusively" not in r.text for r in ChunkStore(index / "chunks.json").all())
    assert all("created exclusively" not in v.document for v in VectorStore(index / "vectors").entries())
    assert all(v.metadata["corpus_admitted"] for v in VectorStore(index / "vectors").entries())


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("legal_status", "under_review", "current"),
        ("legal_status", "repealed", "current"),
        ("rights_status", "unknown", "authorized"),
        ("source_url", "https://pakistancode.gov.pk.evil.test/english", "official host"),
        ("source_url", "http://www.na.gov.pk/en/downloads.php", "official host"),
        ("source_url", "https://user:password@www.na.gov.pk/en/downloads.php", "official host"),
        ("source_url", "https://www.na.gov.pk:8443/en/downloads.php", "official host"),
        ("rights_basis", "", "rights_basis"),
        ("gazette_url", "https://example.com/notice", "official host"),
        ("sha256", "a" * 64, "SHA-256 mismatch"),
    ],
)
def test_unsafe_sources_are_rejected(tmp_path, field, value, expected):
    root, doc, entry, save = fixture_corpus(tmp_path)
    entry[field] = value
    save()
    with pytest.raises(SourceRegistryError, match=expected):
        validate_registry(root)


def test_expired_reviews_are_rejected(tmp_path):
    root, doc, entry, save = fixture_corpus(tmp_path)
    entry["reviewed_on"] = (date.today() - timedelta(days=91)).isoformat()
    save()
    with pytest.raises(SourceRegistryError, match="older than 90"):
        validate_registry(root)


def test_registry_must_cover_every_file(tmp_path):
    root, doc, entry, save = fixture_corpus(tmp_path)
    (root / "unreviewed.txt").write_text("Unknown statute text.", encoding="utf-8")
    with pytest.raises(SourceRegistryError, match="differs from corpus"):
        validate_registry(root)


def test_missing_registry_rejected_but_demo_ingestion_remains_available(tmp_path):
    root, doc, entry, save = fixture_corpus(tmp_path)
    (root / "source_registry.json").unlink()
    with pytest.raises(SourceRegistryError, match="Missing source_registry"):
        run_ingestion(root, tmp_path / "strict", admitted_only=True)
    ingested, embedded = run_ingestion(root, tmp_path / "demo")
    assert ingested > 0 and embedded > 0


def test_production_startup_requires_admitted_metadata(tmp_path, monkeypatch):
    root, doc, entry, save = fixture_corpus(tmp_path)
    index = tmp_path / "index"
    run_ingestion(root, index, admitted_only=True)
    monkeypatch.setenv("INDEX_DIR", str(index))
    monkeypatch.setenv("APP_ENV", "production")
    app = create_app(qa_service=object(), search_service=object(), api_key="test-key")
    assert TestClient(app).get("/ready").status_code == 200

    # A source URL alone does not constitute review or redistribution approval.
    monkeypatch.setenv("APP_ENV", "development")
    (root / "sources.json").write_text(
        json.dumps({doc.name: {
            "title": entry["title"], "source_url": entry["source_url"],
            "basis": "not legal approval",
        }}),
        encoding="utf-8",
    )
    rogue = tmp_path / "rogue"
    run_ingestion(root, rogue)
    assert all(not row.metadata.get("corpus_admitted")
               for row in ChunkStore(rogue / "chunks.json").all())

    monkeypatch.setenv("INDEX_DIR", str(rogue))
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError, match="Production requires"):
        create_app(qa_service=object(), search_service=object(), api_key="test-key")


def test_production_ingest_requires_registry_even_without_cli_flag(tmp_path, monkeypatch):
    root, doc, entry, save = fixture_corpus(tmp_path)
    (root / "source_registry.json").unlink()
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(SourceRegistryError, match="Missing source_registry"):
        run_ingestion(root, tmp_path / "index")

def test_future_review_and_expired_gazette_are_rejected(tmp_path):
    root, doc, entry, save = fixture_corpus(tmp_path)
    entry["reviewed_on"] = (date.today() + timedelta(days=1)).isoformat()
    save()
    with pytest.raises(SourceRegistryError, match="future"):
        validate_registry(root)

    entry["reviewed_on"] = date.today().isoformat()
    entry["gazette_checked_on"] = (date.today() - timedelta(days=91)).isoformat()
    save()
    with pytest.raises(SourceRegistryError, match="Gazette check"):
        validate_registry(root)


def test_symlinked_or_nested_corpus_document_is_rejected(tmp_path):
    root, doc, entry, save = fixture_corpus(tmp_path)
    hidden = tmp_path / "target.txt"
    hidden.write_bytes(doc.read_bytes())
    doc.unlink()
    doc.symlink_to(hidden)
    with pytest.raises(SourceRegistryError, match="symlinked"):
        validate_registry(root)


def test_unreadable_pdf_fails_before_index_changes(tmp_path):
    root, doc, entry, save = fixture_corpus(tmp_path)
    doc.unlink()
    pdf = root / "test_act.pdf"
    pdf.write_bytes(b"not a valid PDF")
    entry["sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    registry = root / "source_registry.json"
    registry.write_text(
        json.dumps({"schema_version": 1, "documents": {pdf.name: entry}}),
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        run_ingestion(root, tmp_path / "index", admitted_only=True)
    assert not (tmp_path / "index" / "chunks.json").exists()
