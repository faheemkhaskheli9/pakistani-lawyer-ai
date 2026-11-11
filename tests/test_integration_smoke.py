"""API smoke tests using an actual ingested local corpus and retriever."""
from __future__ import annotations

from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.ingest import run_ingestion


def test_end_to_end_ingest_search_ask_and_readiness(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "contract_act.txt").write_text(
        "Section 1. Free consent means both parties freely agree to enter a "
        "contract. Contract formation requires free consent of both parties.",
        encoding="utf-8",
    )
    index = tmp_path / "index"
    processed, embedded = run_ingestion(corpus, index)
    assert processed > 0 and embedded > 0

    monkeypatch.setenv("INDEX_DIR", str(index))
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(create_app(api_key="smoke-key"))
    headers = {"X-API-Key": "smoke-key"}

    assert client.get("/health").status_code == 200
    assert client.get("/ready").json() == {"status": "ready"}

    searched = client.get("/api/v1/search", params={"q": "free consent"}, headers=headers)
    assert searched.status_code == 200 and searched.json()["results"]

    answered = client.post(
        "/api/v1/ask",
        json={"question": "What does free consent mean for a contract?"},
        headers=headers,
    )
    assert answered.status_code == 200
    assert answered.json()["status"] == "answered"
    assert answered.json()["answer"]["citations"]

    unrelated = client.post(
        "/api/v1/ask",
        json={"question": "zxqv9 mnp3 zzxyz"},
        headers=headers,
    )
    assert unrelated.status_code == 200
    assert unrelated.json()["status"] == "no_relevant_source"


def test_readiness_fails_after_all_sources_removed(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "law.txt"
    doc.write_text("Section 1. A short law.", encoding="utf-8")
    index = tmp_path / "index"
    run_ingestion(corpus, index)
    doc.unlink()
    run_ingestion(corpus, index)

    monkeypatch.setenv("INDEX_DIR", str(index))
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(create_app(api_key="smoke-key"))
    assert client.get("/ready").status_code == 503
