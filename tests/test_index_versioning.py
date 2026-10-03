import json

from legal_core.ingest import run_ingestion
from legal_core.vector_store import VectorStore


def make_corpus(tmp_path, **files):
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    for name, text in files.items():
        (corpus / name).write_text(text, encoding="utf-8")
    return corpus


def test_manifest_records_build_inputs(tmp_path):
    corpus = make_corpus(tmp_path, **{"a.txt": "Section 1. One\nbody text"})
    run_ingestion(corpus, tmp_path / "idx", embedding_provider="hashing")
    manifest = json.loads((tmp_path / "idx" / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    assert manifest["embedding_provider"] == "hashing"
    assert manifest["embedding_dimension"] == 256
    assert manifest["chunk_count"] == 1 and set(manifest["sources"]) == {"a.txt"}


def test_changed_and_removed_sources_are_pruned(tmp_path):
    corpus = make_corpus(tmp_path, **{"a.txt": "Section 1. Old\nold body", "b.txt": "Section 1. Keep\nkeep body"})
    run_ingestion(corpus, tmp_path / "idx", embedding_provider="hashing")

    (corpus / "a.txt").write_text("Section 1. New\nnew body", encoding="utf-8")
    run_ingestion(corpus, tmp_path / "idx", embedding_provider="hashing")
    docs = [e.document for e in VectorStore(tmp_path / "idx" / "vectors").entries()]
    assert len(docs) == 2 and not any("old body" in d for d in docs)

    (corpus / "b.txt").unlink()
    run_ingestion(corpus, tmp_path / "idx", embedding_provider="hashing")
    docs = [e.document for e in VectorStore(tmp_path / "idx" / "vectors").entries()]
    assert len(docs) == 1 and "new body" in docs[0]


def test_vector_store_delete_handles_missing_and_empty(tmp_path):
    store = VectorStore(tmp_path / "v")
    store.upsert("x", [1.0, 0.0], "doc")
    assert store.delete(["nope"]) == 0
    assert store.delete(["x"]) == 1 and len(store) == 0
    store.upsert("y", [1.0, 0.0, 0.0], "doc2")  # dimension reset after emptying
    assert len(VectorStore(tmp_path / "v")) == 1
