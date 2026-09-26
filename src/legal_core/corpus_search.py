"""No-LLM corpus browse/search service (issue #10).

This module exposes retrieved source text directly and never invokes an LLM.
It is therefore useful both for source-first research and as a fallback when
generation providers are unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .filters import RetrievalFilters
from .retrieval import Retriever, RetrievedChunk
from .safeguards import LEGAL_INFORMATION_DISCLAIMER


@dataclass(frozen=True)
class CorpusSearchResult:
    rank: int
    chunk_id: str
    score: float
    source: str
    section: str | None
    citation: str
    excerpt: str
    source_path: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusSearchResponse:
    query: str
    results: tuple[CorpusSearchResult, ...]
    disclaimer: str = LEGAL_INFORMATION_DISCLAIMER


def _result_from_chunk(chunk: RetrievedChunk) -> CorpusSearchResult | None:
    if not chunk.source:
        return None

    excerpt = " ".join(chunk.text.split())
    if len(excerpt) > 500:
        excerpt = excerpt[:497].rstrip() + "..."

    safe_name = PurePosixPath(chunk.source).name
    return CorpusSearchResult(
        rank=chunk.rank,
        chunk_id=chunk.chunk_id,
        score=chunk.score,
        source=safe_name,
        section=chunk.section,
        citation=chunk.citation,
        excerpt=excerpt,
        source_path=f"data/corpus/{safe_name}",
        metadata=dict(chunk.metadata),
    )


class CorpusSearchService:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> CorpusSearchResponse:
        if not query or not query.strip():
            raise ValueError("query must not be blank")

        if filters is None:
            chunks = self.retriever.retrieve(query.strip(), top_k=top_k)
        else:
            try:
                chunks = self.retriever.retrieve(
                    query.strip(), top_k=top_k, filters=filters
                )
            except TypeError as exc:
                raise ValueError(
                    "configured retriever does not support metadata filters"
                ) from exc
        results = tuple(
            result
            for chunk in chunks
            for result in [_result_from_chunk(chunk)]
            if result is not None
        )
        return CorpusSearchResponse(query=query.strip(), results=results)
