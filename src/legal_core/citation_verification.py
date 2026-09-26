"""Citation integrity checks for grounded legal answers."""
from __future__ import annotations

from dataclasses import dataclass

from .qa import AnswerCitation, LegalAnswer
from .retrieval import RetrievedChunk


@dataclass(frozen=True)
class CitationCheck:
    chunk_id: str
    verified: bool
    reason: str


@dataclass(frozen=True)
class CitationVerificationReport:
    verified: bool
    checks: tuple[CitationCheck, ...]
    total_citations: int
    verified_citations: int


def _matches_chunk(citation: AnswerCitation, chunk: RetrievedChunk) -> tuple[bool, str]:
    if citation.source != chunk.source:
        return False, "source does not match retrieved chunk"
    if citation.section != chunk.section:
        return False, "section does not match retrieved chunk"
    if citation.citation != chunk.citation:
        return False, "rendered citation does not match retrieved chunk"
    return True, "citation matches retrieved grounding chunk"


def verify_citations(
    answer: LegalAnswer,
    chunks: list[RetrievedChunk],
) -> CitationVerificationReport:
    """Verify every exposed citation against the exact retrieved grounding set."""
    grounding = {chunk.chunk_id: chunk for chunk in chunks}
    seen: set[str] = set()
    checks: list[CitationCheck] = []

    for citation in answer.citations:
        if citation.chunk_id in seen:
            checks.append(
                CitationCheck(
                    chunk_id=citation.chunk_id,
                    verified=False,
                    reason="duplicate citation for the same grounding chunk",
                )
            )
            continue
        seen.add(citation.chunk_id)

        chunk = grounding.get(citation.chunk_id)
        if chunk is None:
            checks.append(
                CitationCheck(
                    chunk_id=citation.chunk_id,
                    verified=False,
                    reason="citation does not reference a retrieved grounding chunk",
                )
            )
            continue

        valid, reason = _matches_chunk(citation, chunk)
        checks.append(CitationCheck(citation.chunk_id, valid, reason))

    if not answer.citations:
        checks.append(
            CitationCheck(
                chunk_id="",
                verified=False,
                reason="generated answer contains no verifiable citations",
            )
        )

    verified_count = sum(check.verified for check in checks)
    all_verified = bool(answer.citations) and all(check.verified for check in checks)
    return CitationVerificationReport(
        verified=all_verified,
        checks=tuple(checks),
        total_citations=len(answer.citations),
        verified_citations=verified_count,
    )
