"""End-to-end retrieval gate for safe legal Q&A (issue #9)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .citation_verification import CitationVerificationReport, verify_citations
from .generation import AnswerGenerator
from .filters import RetrievalFilters
from .qa import LegalAnswer, answer_from_chunks
from .query_analysis import QueryAnalysis, analyze_query
from .retrieval import RetrievedChunk, Retriever
from .safeguards import LEGAL_INFORMATION_DISCLAIMER

MIN_SCORE_ENV = "RETRIEVAL_MIN_SCORE"
DEFAULT_MIN_SCORE = 0.20
NO_RELEVANT_SOURCE_MESSAGE = (
    "No relevant source found in the available legal corpus. "
    "Try a more specific question or search the corpus directly."
)


def resolve_min_score(
    min_score: float | None = None,
    *,
    env: dict | None = None,
) -> float:
    """Resolve a cosine-similarity threshold in the inclusive range [0, 1]."""
    if min_score is not None:
        value = min_score
        origin = "min_score"
    else:
        env = os.environ if env is None else env
        raw = env.get(MIN_SCORE_ENV)
        if raw is None or not raw.strip():
            return DEFAULT_MIN_SCORE
        try:
            value = float(raw.strip())
        except ValueError:
            raise ValueError(
                f"$RETRIEVAL_MIN_SCORE must be a number between 0 and 1, got {raw!r}"
            ) from None
        origin = "$RETRIEVAL_MIN_SCORE"

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{origin} must be a number between 0 and 1, got {value!r}")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{origin} must be between 0 and 1, got {value!r}")
    return value


@dataclass(frozen=True)
class QAResult:
    status: str
    message: str
    answer: LegalAnswer | None
    query_analysis: QueryAnalysis | None = None
    citation_verification: CitationVerificationReport | None = None
    disclaimer: str = LEGAL_INFORMATION_DISCLAIMER

    @property
    def found_relevant_source(self) -> bool:
        return self.status == "answered"


def _merge_ranked_chunks(groups: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Deduplicate ranked groups while preserving the best score per chunk."""
    by_id: dict[str, RetrievedChunk] = {}
    order: dict[str, int] = {}
    counter = 0
    for group in groups:
        for chunk in group:
            if chunk.chunk_id not in order:
                order[chunk.chunk_id] = counter
                counter += 1
            existing = by_id.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                by_id[chunk.chunk_id] = chunk

    ranked = sorted(
        by_id.values(),
        key=lambda chunk: (-chunk.score, order[chunk.chunk_id]),
    )
    return [
        RetrievedChunk(
            rank=rank,
            chunk_id=chunk.chunk_id,
            score=chunk.score,
            text=chunk.text,
            source=chunk.source,
            section=chunk.section,
            chunk_index=chunk.chunk_index,
            metadata=dict(chunk.metadata),
        )
        for rank, chunk in enumerate(ranked, start=1)
    ]


class QuestionAnsweringService:
    """Retrieve, apply a relevance floor, then generate only when grounded."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        generator: AnswerGenerator | None = None,
        min_score: float | None = None,
    ):
        self.retriever = retriever
        self.generator = generator or AnswerGenerator()
        self.min_score = resolve_min_score(min_score)

    def _retrieve_for_analysis(self, analysis: QueryAnalysis) -> list[RetrievedChunk]:
        """Use jurisdiction hints when supported, with federal and broad fallback."""
        jurisdiction = analysis.jurisdiction
        if not jurisdiction:
            return self.retriever.retrieve(analysis.normalized_query)

        groups: list[list[RetrievedChunk]] = []
        try:
            groups.append(
                self.retriever.retrieve(
                    analysis.normalized_query,
                    filters=RetrievalFilters(jurisdiction=jurisdiction),
                )
            )
            if jurisdiction != "Federal":
                groups.append(
                    self.retriever.retrieve(
                        analysis.normalized_query,
                        filters=RetrievalFilters(jurisdiction="Federal"),
                    )
                )
        except TypeError:
            # Backward compatibility for simple/custom retrievers that do not
            # yet expose the metadata-filter keyword.
            return self.retriever.retrieve(analysis.normalized_query)

        combined = _merge_ranked_chunks(groups)
        if combined:
            return combined
        return self.retriever.retrieve(analysis.normalized_query)

    def ask(self, question: str) -> QAResult:
        if not question or not question.strip():
            raise ValueError("question must not be blank")

        analysis = analyze_query(question)
        retrieved = self._retrieve_for_analysis(analysis)
        relevant = [chunk for chunk in retrieved if chunk.score >= self.min_score]

        if not relevant:
            return QAResult(
                status="no_relevant_source",
                message=NO_RELEVANT_SOURCE_MESSAGE,
                answer=None,
                query_analysis=analysis,
            )

        answer = answer_from_chunks(
            question.strip(),
            relevant,
            generator=self.generator,
        )
        verification = verify_citations(answer, relevant)
        if not verification.verified:
            return QAResult(
                status="citation_verification_failed",
                message=(
                    "The generated answer was withheld because one or more "
                    "citations could not be verified against retrieved sources."
                ),
                answer=None,
                query_analysis=analysis,
                citation_verification=verification,
            )
        return QAResult(
            status="answered",
            message="Answer generated from retrieved legal sources.",
            answer=answer,
            query_analysis=analysis,
            citation_verification=verification,
        )
