import json
import logging

import pytest
from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.security import client_identity, resolve_api_key, resolve_trusted_proxies
from tests.test_api import QAService, SearchService


def test_production_rejects_default_or_missing_key():
    with pytest.raises(ValueError):
        resolve_api_key(env={"PAKISTANI_LAWYER_ENV": "production"})
    with pytest.raises(ValueError):
        resolve_api_key(env={"PAKISTANI_LAWYER_ENV": "prod", "PAKISTANI_LAWYER_API_KEY": "local-development-key"})
    assert resolve_api_key(env={"PAKISTANI_LAWYER_ENV": "prod", "PAKISTANI_LAWYER_API_KEY": "s3cret"}) == "s3cret"
    assert resolve_api_key(env={}) == "local-development-key"  # dev stays frictionless


def test_client_identity_ignores_spoofable_forwarded_for_by_default():
    assert client_identity("10.0.0.1", "1.2.3.4") == "10.0.0.1"
    assert client_identity("10.0.0.1", "spoofed, 1.2.3.4", trusted_proxies=1) == "1.2.3.4"
    assert client_identity("10.0.0.1", "1.2.3.4", trusted_proxies=2) == "10.0.0.1"
    assert client_identity(None, None) == "unknown"


def test_trusted_proxy_count_validation():
    assert resolve_trusted_proxies(env={}) == 0
    assert resolve_trusted_proxies(env={"TRUSTED_PROXY_COUNT": "2"}) == 2
    with pytest.raises(ValueError):
        resolve_trusted_proxies(env={"TRUSTED_PROXY_COUNT": "-1"})


def make(**kw):
    return TestClient(create_app(qa_service=QAService(), search_service=SearchService(), api_key="k", **kw))


def test_search_is_rate_limited():
    c = make(search_rate_limit=2)
    assert [c.get("/api/v1/search", params={"q": "x"}).status_code for _ in range(3)] == [200, 200, 429]


def test_rate_limit_keys_on_forwarded_client_only_when_trusted():
    c = make(qa_rate_limit=1, trusted_proxies=1)

    def post(ip):
        headers = {"X-API-Key": "k", "X-Forwarded-For": ip}
        return c.post("/api/v1/ask", json={"question": "q"}, headers=headers).status_code

    assert post("1.1.1.1") == 200 and post("2.2.2.2") == 200 and post("1.1.1.1") == 429


def test_access_log_has_request_id_and_never_logs_query(caplog):
    c = make()
    with caplog.at_level(logging.INFO, logger="legal_core.access"):
        r = c.get("/api/v1/search", params={"q": "very-secret-client-matter"}, headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    record = json.loads(caplog.records[-1].message)
    assert record["request_id"] == "abc123" and record["status"] == 200 and record["latency_ms"] >= 0
    assert "very-secret-client-matter" not in caplog.text


def test_default_services_honor_index_dir_env(tmp_path, monkeypatch):
    from legal_core import api
    from legal_core.ingest import run_ingestion

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Section 1. Bail\nan accused person may apply for bail", encoding="utf-8")
    run_ingestion(corpus, tmp_path / "idx", embedding_provider="hashing")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    qa, search = api._default_services()
    assert len(search.search("bail").results) >= 1
