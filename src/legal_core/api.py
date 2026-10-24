"""FastAPI application exposing the legal research services."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .corpus_search import CorpusSearchService
from .filters import RetrievalFilters
from .generation import GenerationError
from .hybrid_retrieval import build_hybrid_retriever
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
    api_key: str | None = None,
    qa_rate_limit: int | None = None,
    trust_proxy: bool | None = None,
) -> FastAPI:
    expected_api_key = resolve_api_key(api_key)  # fail fast when no key is configured
    if qa_service is None or search_service is None:
        default_qa, default_search = _default_services()
        qa_service = qa_service or default_qa
        search_service = search_service or default_search

    limiter = SlidingWindowRateLimiter(resolve_rate_limit(qa_rate_limit))

    app = FastAPI(
        title="Pakistani Lawyer AI",
        version="0.1.0",
        description="Citation-grounded legal information and corpus search API.",
    )

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
