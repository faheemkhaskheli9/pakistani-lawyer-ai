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
    def search(self, query, top_k=None):
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
