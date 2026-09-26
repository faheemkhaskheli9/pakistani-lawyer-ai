"""FastAPI application exposing the legal research services."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .corpus_search import CorpusSearchService
from .hybrid_retrieval import build_hybrid_retriever
from .security import FixedWindowRateLimiter, api_key_matches, resolve_api_key, resolve_rate_limit
from .service import QuestionAnsweringService


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)


def _default_services():
    retriever = build_hybrid_retriever()
    return QuestionAnsweringService(retriever), CorpusSearchService(retriever)


def create_app(
    *,
    qa_service: Any | None = None,
    search_service: Any | None = None,
    api_key: str | None = None,
    qa_rate_limit: int | None = None,
) -> FastAPI:
    if qa_service is None or search_service is None:
        default_qa, default_search = _default_services()
        qa_service = qa_service or default_qa
        search_service = search_service or default_search

    expected_api_key = resolve_api_key(api_key)
    limiter = FixedWindowRateLimiter(resolve_rate_limit(qa_rate_limit))

    app = FastAPI(
        title="Pakistani Lawyer AI",
        version="0.1.0",
        description="Citation-grounded legal information and corpus search API.",
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/v1/ask")
    def ask(
        payload: AskRequest,
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        if not api_key_matches(x_api_key, expected_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        client_ip = request.client.host if request.client else "unknown"
        if not limiter.allow(client_ip):
            raise HTTPException(
                status_code=429,
                detail="QA rate limit exceeded",
                headers={"Retry-After": "60"},
            )

        try:
            result = qa_service.ask(payload.question)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(result)

    @app.get("/api/v1/search")
    def search(
        q: str = Query(min_length=1, max_length=5000),
        top_k: int | None = Query(default=None, ge=1, le=100),
    ):
        try:
            result = search_service.search(q, top_k=top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(result)

    return app


app = create_app()
