"""FastAPI application exposing the legal research services."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .browse import CorpusBrowseService
from .corpus_search import CorpusSearchService
from .filters import RetrievalFilters
from .generation import GenerationError
from .hybrid_retrieval import build_hybrid_retriever
from .limitation import RULES, calculate_limitation
from .security import (
    SlidingWindowRateLimiter,
    api_key_matches,
    client_identity,
    resolve_api_key,
    resolve_rate_limit,
)
from .service import QuestionAnsweringService


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)


STREAM_CHUNK_WORDS = 8
WEB_UI_PATH = Path(__file__).parent / "web" / "index.html"


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_events(result):
    """Yield SSE frames for a finished QA result.

    The answer is only streamed after citation verification has passed, so
    unverified model output is never sent to the client.
    """
    payload = asdict(result)
    answer = payload.get("answer")
    yield _sse("status", {"status": payload["status"], "message": payload["message"]})
    if answer:
        words = answer["answer"].split(" ")
        for i in range(0, len(words), STREAM_CHUNK_WORDS):
            text = " ".join(words[i : i + STREAM_CHUNK_WORDS])
            yield _sse("delta", {"text": text if i == 0 else " " + text})
        yield _sse("citations", {"citations": answer["citations"]})
    yield _sse("done", {"status": payload["status"], "disclaimer": payload["disclaimer"]})


def _default_services():
    retriever = build_hybrid_retriever()
    return QuestionAnsweringService(retriever), CorpusSearchService(retriever)


def create_app(
    *,
    qa_service: Any | None = None,
    search_service: Any | None = None,
    browse_service: Any | None = None,
    api_key: str | None = None,
    qa_rate_limit: int | None = None,
    trust_proxy: bool | None = None,
) -> FastAPI:
    expected_api_key = resolve_api_key(api_key)  # fail fast when no key is configured
    if qa_service is None or search_service is None:
        default_qa, default_search = _default_services()
        qa_service = qa_service or default_qa
        search_service = search_service or default_search

    browse_service = browse_service or CorpusBrowseService()

    limiter = SlidingWindowRateLimiter(resolve_rate_limit(qa_rate_limit))

    app = FastAPI(
        title="Pakistani Lawyer AI",
        version="0.1.0",
        description="Citation-grounded legal information and corpus search API.",
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def web_ui():
        """Minimal web UI; it calls the API with a user-supplied key."""
        return HTMLResponse(WEB_UI_PATH.read_text(encoding="utf-8"))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    def guard(request: Request, x_api_key: str | None) -> None:
        if not api_key_matches(x_api_key, expected_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        if not limiter.allow(client_identity(request, trust_proxy=trust_proxy)):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"},
            )

    @app.post("/api/v1/ask")
    def ask(
        payload: AskRequest,
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        guard(request, x_api_key)
        try:
            result = qa_service.ask(payload.question)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GenerationError as exc:
            raise HTTPException(status_code=502, detail="Answer generation failed") from exc
        return asdict(result)

    @app.post("/api/v1/ask/stream")
    def ask_stream(
        payload: AskRequest,
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        """Server-Sent Events variant of ``/ask`` (verified answers only)."""
        guard(request, x_api_key)
        try:
            result = qa_service.ask(payload.question)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GenerationError as exc:
            raise HTTPException(status_code=502, detail="Answer generation failed") from exc
        return StreamingResponse(
            _stream_events(result),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        guard(request, x_api_key)
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

    @app.get("/api/v1/corpus")
    def corpus_list(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        """Browse the ingested documents (no retrieval, no LLM)."""
        guard(request, x_api_key)
        return {"documents": [asdict(d) for d in browse_service.list_documents()]}

    @app.get("/api/v1/corpus/{source}")
    def corpus_document(
        source: str,
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        guard(request, x_api_key)
        detail = browse_service.get_document(source)
        if detail is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return asdict(detail)

    @app.get("/api/v1/limitation")
    def limitation(
        request: Request,
        kind: str = Query(...),
        start: date = Query(..., description="Date the cause of action accrued (YYYY-MM-DD)"),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        """Limitation-period calculator for selected Limitation Act, 1908 articles."""
        guard(request, x_api_key)
        try:
            result = calculate_limitation(kind, start)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(result)

    @app.get("/api/v1/limitation/rules")
    def limitation_rules(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        guard(request, x_api_key)
        return {"rules": [asdict(r) for r in RULES.values()]}

    return app


_app: FastAPI | None = None


def __getattr__(name: str):
    """Build ``app`` lazily so importing this module has no side effects."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
