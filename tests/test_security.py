"""Authentication and rate-limit tests for issue #12."""
from __future__ import annotations

from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.corpus_search import CorpusSearchResponse
from legal_core.service import QAResult


class QA:
    def ask(self, question):
        return QAResult(status="no_relevant_source", message="No source.", answer=None)


class Search:
    def search(self, query, top_k=None, filters=None):
        return CorpusSearchResponse(query=query, results=())


def make_client(limit=2):
    return TestClient(
        create_app(
            qa_service=QA(),
            search_service=Search(),
            api_key="test-key",
            qa_rate_limit=limit,
        )
    )


def test_qa_endpoint_requires_api_key():
    response = make_client().post("/api/v1/ask", json={"question": "test"})
    assert response.status_code == 401


def test_wrong_api_key_is_rejected():
    response = make_client().post(
        "/api/v1/ask",
        json={"question": "test"},
        headers={"X-API-Key": "wrong"},
    )
    assert response.status_code == 401


def test_requests_beyond_per_ip_limit_return_429():
    client = make_client(limit=2)
    kwargs = {
        "json": {"question": "test"},
        "headers": {"X-API-Key": "test-key"},
    }
    assert client.post("/api/v1/ask", **kwargs).status_code == 200
    assert client.post("/api/v1/ask", **kwargs).status_code == 200
    blocked = client.post("/api/v1/ask", **kwargs)
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "60"


def test_search_is_rate_limited_and_shares_budget_with_ask():
    client = make_client(limit=1)
    headers = {"X-API-Key": "test-key"}
    assert client.get("/api/v1/search", params={"q": "x"}, headers=headers).status_code == 200
    assert client.get("/api/v1/search", params={"q": "x"}, headers=headers).status_code == 429


def test_missing_api_key_fails_fast(monkeypatch):
    import pytest

    monkeypatch.delenv("PAKISTANI_LAWYER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        create_app(qa_service=QA(), search_service=Search())


def test_generation_error_maps_to_502():
    from legal_core.generation import GenerationError

    class Boom:
        def ask(self, question):
            raise GenerationError("provider down")

    client = TestClient(create_app(qa_service=Boom(), search_service=Search(), api_key="k"))
    response = client.post("/api/v1/ask", json={"question": "q"}, headers={"X-API-Key": "k"})
    assert response.status_code == 502
    assert "provider down" not in response.text


def test_forwarded_for_only_honoured_when_proxy_trusted():
    headers = {"X-API-Key": "k"}

    def run(trust):
        client = TestClient(
            create_app(qa_service=QA(), search_service=Search(), api_key="k",
                       qa_rate_limit=1, trust_proxy=trust)
        )
        codes = []
        for ip in ("1.1.1.1", "2.2.2.2"):
            r = client.post("/api/v1/ask", json={"question": "q"},
                            headers={**headers, "X-Forwarded-For": ip})
            codes.append(r.status_code)
        return codes

    assert run(True) == [200, 200]
    assert run(False) == [200, 429]


def test_rate_limiter_evicts_idle_buckets():
    from legal_core.security import SlidingWindowRateLimiter

    now = [0.0]
    limiter = SlidingWindowRateLimiter(5, window_seconds=10, clock=lambda: now[0])
    for i in range(300):
        limiter.allow(f"ip{i}")
    now[0] = 100.0
    for i in range(256):
        limiter.allow("fresh")
    assert len(limiter._hits) < 50
