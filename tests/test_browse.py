"""Corpus browse endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.browse import CorpusBrowseService
from legal_core.chunk_store import ChunkRecord, ChunkStore

KEY = {"X-API-Key": "test-key"}


def make_client(tmp_path):
    ChunkStore(tmp_path / "chunks.json").upsert(
        [
            ChunkRecord("a1", "/x/act.txt", "Section 2", 1, "second", {"jurisdiction": "federal"}),
            ChunkRecord("a0", "/x/act.txt", "Section 1", 0, "first", {"jurisdiction": "federal"}),
            ChunkRecord("b0", "/x/rule.txt", "Rule 1", 0, "rule", {}),
        ]
    )
    return TestClient(
        create_app(
            qa_service=object(),
            search_service=object(),
            browse_service=CorpusBrowseService(tmp_path),
            api_key="test-key",
        )
    )


def test_list_documents(tmp_path):
    body = make_client(tmp_path).get("/api/v1/corpus", headers=KEY).json()
    assert [d["source"] for d in body["documents"]] == ["act.txt", "rule.txt"]
    assert body["documents"][0]["chunks"] == 2


def test_document_detail_is_ordered(tmp_path):
    body = make_client(tmp_path).get("/api/v1/corpus/act.txt", headers=KEY).json()
    assert [s["section"] for s in body["sections"]] == ["Section 1", "Section 2"]


def test_unknown_document_404_and_auth_required(tmp_path):
    c = make_client(tmp_path)
    assert c.get("/api/v1/corpus/nope.txt", headers=KEY).status_code == 404
    assert c.get("/api/v1/corpus").status_code == 401


def test_missing_index_lists_empty(tmp_path):
    assert CorpusBrowseService(tmp_path / "none").list_documents() == ()
