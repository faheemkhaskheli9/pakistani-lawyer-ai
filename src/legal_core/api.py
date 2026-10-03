"""FastAPI application exposing the legal research services."""
from __future__ import annotations

import json
import os
import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .corpus_search import CorpusSearchService
from .filters import RetrievalFilters
from .hybrid_retrieval import build_hybrid_retriever
from .retrieval import DEFAULT_INDEX_DIR
from .security import (
    DEFAULT_SEARCH_RATE_LIMIT,
    SEARCH_RATE_LIMIT_ENV,
    FixedWindowRateLimiter,
    api_key_matches,
    client_identity,
    resolve_api_key,
    resolve_rate_limit,
    resolve_trusted_proxies,
)
from .service import QuestionAnsweringService


access_log = logging.getLogger("legal_core.access")
INDEX_HTML = Path(__file__).parent / "static" / "index.html"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)


def _default_services():
    index_dir = os.environ.get("INDEX_DIR") or DEFAULT_INDEX_DIR
    retriever = build_hybrid_retriever(index_dir)
    if len(retriever.vector_store) == 0:
        access_log.warning(
            "Search index at %s is empty or missing; every question will return "
            "'no relevant source'. Run: python -m legal_core.ingest --corpus data/corpus --index-dir %s",
            index_dir,
            index_dir,
        )
    return QuestionAnsweringService(retriever), CorpusSearchService(retriever)


def create_app(
    *,
    qa_service: Any | None = None,
    search_service: Any | None = None,
    api_key: str | None = None,
    qa_rate_limit: int | None = None,
    search_rate_limit: int | None = None,
    trusted_proxies: int | None = None,
) -> FastAPI:
    if qa_service is None or search_service is None:
        default_qa, default_search = _default_services()
        qa_service = qa_service or default_qa
        search_service = search_service or default_search

    expected_api_key = resolve_api_key(api_key)
    limiter = FixedWindowRateLimiter(resolve_rate_limit(qa_rate_limit))
    search_limiter = FixedWindowRateLimiter(
        resolve_rate_limit(
            search_rate_limit, env_var=SEARCH_RATE_LIMIT_ENV, default=DEFAULT_SEARCH_RATE_LIMIT
        )
    )
    proxies = resolve_trusted_proxies(trusted_proxies)

    def identify(request: Request) -> str:
        peer = request.client.host if request.client else None
        return client_identity(peer, request.headers.get("x-forwarded-for"), proxies)

    app = FastAPI(
        title="Pakistani Lawyer AI",
        version="0.1.0",
        description="Citation-grounded legal information and corpus search API.",
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        """Structured access log + request id. Query text is never logged."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            access_log.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        "client": identify(request),
                    }
                )
            )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))

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

        if not limiter.allow(identify(request)):
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
        request: Request,
        q: str = Query(min_length=1, max_length=5000),
        top_k: int | None = Query(default=None, ge=1, le=100),
        jurisdiction: str | None = Query(default=None),
        court: str | None = Query(default=None),
        document_type: str | None = Query(default=None),
        case_citation: str | None = Query(default=None),
        date_from: str | None = Query(default=None),
        date_to: str | None = Query(default=None),
    ):
        if not search_limiter.allow(identify(request)):
            raise HTTPException(
                status_code=429,
                detail="Search rate limit exceeded",
                headers={"Retry-After": "60"},
            )
        try:
            filters = RetrievalFilters(
                jurisdiction=jurisdiction,
                court=court,
                document_type=document_type,
                case_citation=case_citation,
                date_from=date_from,
                date_to=date_to,
            )
            result = search_service.search(q, top_k=top_k, filters=filters)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(result)

    return app


app = create_app()
