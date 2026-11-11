"""API integration tests for health, Q&A and direct corpus search."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.corpus_search import CorpusSearchResponse, CorpusSearchResult
from legal_core.qa import AnswerCitation, LegalAnswer
from legal_core.safeguards import LEGAL_INFORMATION_DISCLAIMER
from legal_core.service import QAResult

TEST_API_KEY = "test-key"


class QAService:
    def ask(self, question):
        return QAResult(
            status="answered",
            message="Answer generated from retrieved legal sources.",
            answer=LegalAnswer(
                question=question,
                answer="Grounded response.",
                citations=(
                    AnswerCitation(
                        chunk_id="c1",
                        source="sample.txt",
                        section="Section 1",
                        citation="sample.txt, Section 1",
                        score=0.9,
                        excerpt="Source excerpt.",
                    ),
                ),
            ),
        )


class SearchService:
    def search(self, query, top_k=None, filters=None):
        return CorpusSearchResponse(
            query=query,
            results=(
                CorpusSearchResult(
                    rank=1,
                    chunk_id="c1",
                    score=0.9,
                    source="sample.txt",
                    section="Section 1",
                    citation="sample.txt, Section 1",
                    excerpt="Source excerpt.",
                    source_path="data/corpus/sample.txt",
                ),
            ),
        )


def client(rate_limit=30):
    return TestClient(
        create_app(
            qa_service=QAService(),
            search_service=SearchService(),
            api_key=TEST_API_KEY,
            qa_rate_limit=rate_limit,
        )
    )


def test_health_endpoint():
    assert client().get("/health").json() == {"status": "ok"}


def test_ask_endpoint_returns_grounded_answer_and_citations():
    response = client().post(
        "/api/v1/ask",
        json={"question": "What does the source say?"},
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"]["citations"][0]["citation"] == "sample.txt, Section 1"
    assert body["disclaimer"] == LEGAL_INFORMATION_DISCLAIMER


def test_search_endpoint_requires_api_key():
    assert client().get("/api/v1/search", params={"q": "source"}).status_code == 401


def test_search_endpoint_works_with_api_key_and_no_llm():
    response = client().get(
        "/api/v1/search",
        params={"q": "source", "top_k": 3},
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["source_path"] == "data/corpus/sample.txt"


def _parse_sse(text):
    events = []
    for frame in text.strip().split("\n\n"):
        lines = frame.split("\n")
        events.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return events


def test_ask_stream_emits_status_deltas_citations_and_done():
    response = client().post(
        "/api/v1/ask/stream",
        json={"question": "What does the source say?"},
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "status" and names[-1] == "done"
    assert "".join(d["text"] for n, d in events if n == "delta") == "Grounded response."
    citations = dict(events)["citations"]["citations"]
    assert citations[0]["citation"] == "sample.txt, Section 1"
    assert dict(events)["done"]["disclaimer"] == LEGAL_INFORMATION_DISCLAIMER


def test_ask_stream_requires_api_key():
    response = client().post("/api/v1/ask/stream", json={"question": "x"})
    assert response.status_code == 401


def test_web_ui_served_with_disclaimer_rtl_and_safe_rendering():
    client = TestClient(create_app(qa_service=QAService(), search_service=SearchService(), api_key=TEST_API_KEY))
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "not legal advice" in html
    assert "dir" in html and "rtl" in html
    assert "innerHTML" not in html  # model/corpus text must be rendered as text
    assert "sessionStorage" not in html  # never persist sensitive history or API keys

def test_custom_index_dir_is_used_for_corpus_browsing(tmp_path, monkeypatch):
    from legal_core.chunk_store import ChunkRecord, ChunkStore

    index = tmp_path / "index"
    ChunkStore(index / "chunks.json").upsert([
        ChunkRecord(id="custom", source="law.txt", section="Section 4", chunk_index=0,
                    text="Section 4. Current law.")
    ])
    monkeypatch.setenv("INDEX_DIR", str(index))
    api = TestClient(create_app(qa_service=QAService(), search_service=SearchService(), api_key=TEST_API_KEY))
    response = api.get("/api/v1/corpus", headers={"X-API-Key": TEST_API_KEY})
    assert response.status_code == 200
    assert response.json()["documents"][0]["source"] == "law.txt"

def test_readiness_rejects_empty_index(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "missing"))
    c = TestClient(create_app(qa_service=QAService(), search_service=SearchService(), api_key=TEST_API_KEY))
    r = c.get("/ready")
    assert r.status_code == 503


def test_production_refuses_fictional_seed_documents(tmp_path, monkeypatch):
    import pytest

    from legal_core.chunk_store import ChunkRecord, ChunkStore

    index = tmp_path / "index"
    ChunkStore(index / "chunks.json").upsert([
        ChunkRecord(id="fiction", source="sample_civil_code.txt", section="Section 1",
                    chunk_index=0, text="Sample text.", metadata={"source_url": None})
    ])
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("INDEX_DIR", str(index))
    with pytest.raises(RuntimeError, match="Production requires"):
        create_app(qa_service=QAService(), search_service=SearchService(), api_key=TEST_API_KEY)
