"""API integration tests for health, Q&A and direct corpus search."""
from __future__ import annotations

from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.corpus_search import CorpusSearchResponse, CorpusSearchResult
from legal_core.qa import AnswerCitation, LegalAnswer
from legal_core.safeguards import LEGAL_INFORMATION_DISCLAIMER
from legal_core.service import QAResult


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
    def search(self, query, top_k=None):
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


def client():
    return TestClient(create_app(qa_service=QAService(), search_service=SearchService()))


def test_health_endpoint():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_returns_grounded_answer_and_citations():
    response = client().post("/api/v1/ask", json={"question": "What does the source say?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"]["answer"] == "Grounded response."
    assert body["answer"]["citations"][0]["citation"] == "sample.txt, Section 1"
    assert body["disclaimer"] == LEGAL_INFORMATION_DISCLAIMER


def test_search_endpoint_returns_direct_corpus_results():
    response = client().get("/api/v1/search", params={"q": "source", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "source"
    assert body["results"][0]["source_path"] == "data/corpus/sample.txt"
    assert body["disclaimer"] == LEGAL_INFORMATION_DISCLAIMER


def test_request_validation_rejects_blank_question_and_invalid_top_k():
    assert client().post("/api/v1/ask", json={"question": ""}).status_code == 422
    assert client().get("/api/v1/search", params={"q": "x", "top_k": 0}).status_code == 422
