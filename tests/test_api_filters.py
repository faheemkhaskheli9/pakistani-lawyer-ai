"""API coverage for metadata search filters."""
from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.corpus_search import CorpusSearchResponse
from legal_core.service import QAResult


class QA:
    def ask(self, question):
        return QAResult(status="no_relevant_source", message="No source.", answer=None)


class Search:
    def __init__(self):
        self.filters = None

    def search(self, query, top_k=None, filters=None):
        self.filters = filters
        return CorpusSearchResponse(query=query, results=())


def test_search_api_builds_metadata_filters():
    search = Search()
    client = TestClient(
        create_app(
            qa_service=QA(),
            search_service=search,
            api_key="test-key",
            qa_rate_limit=30,
        )
    )
    response = client.get(
        "/api/v1/search",
        params={
            "q": "constitutional",
            "jurisdiction": "Sindh",
            "court": "Sindh High Court",
            "document_type": "judgment",
            "date_from": "2025-01-01",
            "date_to": "2025-12-31",
        },
    )
    assert response.status_code == 200
    assert search.filters.jurisdiction == "Sindh"
    assert search.filters.court == "Sindh High Court"
    assert search.filters.document_type == "judgment"
    assert search.filters.date_from == "2025-01-01"


def test_search_api_rejects_invalid_date_filter():
    client = TestClient(
        create_app(
            qa_service=QA(),
            search_service=Search(),
            api_key="test-key",
            qa_rate_limit=30,
        )
    )
    response = client.get(
        "/api/v1/search",
        params={"q": "x", "date_from": "not-a-date"},
    )
    assert response.status_code == 400
