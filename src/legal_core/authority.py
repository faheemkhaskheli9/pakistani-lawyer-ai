"""Deterministic legal-authority scoring for retrieval reranking.

Authority is intentionally a small bounded signal. It never replaces text
relevance and only helps distinguish otherwise-close legal sources.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthorityAssessment:
    score: float
    reason: str


def assess_authority(metadata: dict) -> AuthorityAssessment:
    document_type = str(metadata.get("document_type") or "").strip().casefold()
    court = str(metadata.get("court") or "").strip().casefold()

    if document_type in {"statute", "constitution", "ordinance", "rules", "regulation"}:
        return AuthorityAssessment(1.0, "primary legislation")

    if "supreme court" in court or court == "sc":
        return AuthorityAssessment(1.0, "Supreme Court judgment")

    if "federal shariat court" in court or court == "fsc":
        return AuthorityAssessment(0.95, "Federal Shariat Court judgment")

    if "high court" in court:
        return AuthorityAssessment(0.90, "High Court judgment")

    if document_type == "judgment":
        return AuthorityAssessment(0.78, "other judgment")

    if document_type in {"commentary", "secondary", "article", "note"}:
        return AuthorityAssessment(0.55, "secondary legal source")

    return AuthorityAssessment(0.70, "unknown or unspecified authority")


def combine_relevance_and_authority(
    relevance_score: float,
    authority: AuthorityAssessment,
    *,
    authority_weight: float = 0.10,
) -> float:
    if not 0.0 <= authority_weight <= 1.0:
        raise ValueError("authority_weight must be between 0 and 1")

    relevance = min(max(float(relevance_score), 0.0), 1.0)
    authority_score = min(max(float(authority.score), 0.0), 1.0)
    return (
        (1.0 - authority_weight) * relevance
        + authority_weight * authority_score
    )
