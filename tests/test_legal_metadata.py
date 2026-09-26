"""Tests for structured legal metadata ingestion (issue #18)."""
from __future__ import annotations

import json

import pytest

from legal_core.chunk_store import ChunkStore
from legal_core.embedding_pipeline import embed_new_chunks
from legal_core.embeddings import HashingEmbedder
from legal_core.ingest import IngestionPipeline
from legal_core.metadata import infer_document_type, load_legal_metadata
from legal_core.retrieval import Retriever
from legal_core.vector_store import VectorStore


def test_metadata_is_optional_and_document_type_is_inferred(tmp_path):
    path = tmp_path / "demo_act.txt"
    path.write_text("Demo Act\nSection 1. Application.", encoding="utf-8")

    metadata = load_legal_metadata(path, text=path.read_text())

    assert metadata.title == "demo_act"
    assert metadata.document_type == "statute"
    assert metadata.jurisdiction is None


def test_sources_and_legal_metadata_manifests_are_merged(tmp_path):
    path = tmp_path / "case.txt"
    path.write_text("Judgment in Example v State.", encoding="utf-8")
    (tmp_path / "sources.json").write_text(
        json.dumps(
            {
                "case.txt": {
                    "title": "Example v State",
                    "source_url": "https://example.invalid/case",
                    "basis": "test fixture",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "legal_metadata.json").write_text(
        json.dumps(
            {
                "case.txt": {
                    "document_type": "judgment",
                    "jurisdiction": "Pakistan",
                    "court": "Supreme Court",
                    "case_citation": "PLD 2026 SC 1",
                    "decision_date": "2026-01-15",
                }
            }
        ),
        encoding="utf-8",
    )

    metadata = load_legal_metadata(path, text=path.read_text())

    assert metadata.title == "Example v State"
    assert metadata.source_url == "https://example.invalid/case"
    assert metadata.document_type == "judgment"
    assert metadata.jurisdiction == "Pakistan"
    assert metadata.court == "Supreme Court"
    assert metadata.case_citation == "PLD 2026 SC 1"
    assert metadata.decision_date == "2026-01-15"


def test_invalid_metadata_date_is_rejected(tmp_path):
    path = tmp_path / "case.txt"
    path.write_text("Judgment", encoding="utf-8")
    (tmp_path / "legal_metadata.json").write_text(
        json.dumps({"case.txt": {"decision_date": "15/01/2026"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        load_legal_metadata(path, text="Judgment")


def test_metadata_propagates_from_ingestion_to_retrieval(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "case.txt"
    source.write_text(
        "Judgment\nSection 1. Constitutional jurisdiction applies.",
        encoding="utf-8",
    )
    (corpus / "legal_metadata.json").write_text(
        json.dumps(
            {
                "case.txt": {
                    "document_type": "judgment",
                    "jurisdiction": "Pakistan",
                    "court": "Supreme Court",
                    "case_citation": "PLD 2026 SC 1",
                }
            }
        ),
        encoding="utf-8",
    )

    chunk_store = ChunkStore(tmp_path / "index" / "chunks.json")
    IngestionPipeline(chunk_store).ingest_file(source)
    record = chunk_store.all()[0]
    assert record.metadata["court"] == "Supreme Court"

    vector_store = VectorStore(tmp_path / "index" / "vectors")
    embedder = HashingEmbedder(dimension=64)
    embed_new_chunks(chunk_store, embedder, vector_store)

    result = Retriever(embedder, vector_store, top_k=1).retrieve(
        "constitutional jurisdiction"
    )[0]
    assert result.metadata["document_type"] == "judgment"
    assert result.metadata["jurisdiction"] == "Pakistan"
    assert result.metadata["court"] == "Supreme Court"
    assert result.metadata["case_citation"] == "PLD 2026 SC 1"


def test_document_type_inference_recognizes_judgment():
    assert infer_document_type(
        __import__("pathlib").Path("decision.txt"),
        "Final Judgment in the matter",
    ) == "judgment"
