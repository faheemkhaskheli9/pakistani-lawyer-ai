"""Structured metadata filters for legal retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class RetrievalFilters:
    jurisdiction: str | None = None
    court: str | None = None
    document_type: str | None = None
    case_citation: str | None = None
    date_from: str | None = None
    date_to: str | None = None

    def __post_init__(self):
        for field_name in ("date_from", "date_to"):
            value = getattr(self, field_name)
            if value is not None:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    raise ValueError(f"{field_name} must use YYYY-MM-DD format") from None
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")

    @property
    def active(self) -> bool:
        return any(
            value is not None
            for value in (
                self.jurisdiction,
                self.court,
                self.document_type,
                self.case_citation,
                self.date_from,
                self.date_to,
            )
        )


def _equal_ci(actual, expected: str | None) -> bool:
    if expected is None:
        return True
    if actual is None:
        return False
    return str(actual).strip().casefold() == expected.strip().casefold()


def matches_filters(metadata: dict, filters: RetrievalFilters | None) -> bool:
    if filters is None or not filters.active:
        return True

    if not _equal_ci(metadata.get("jurisdiction"), filters.jurisdiction):
        return False
    if not _equal_ci(metadata.get("court"), filters.court):
        return False
    if not _equal_ci(metadata.get("document_type"), filters.document_type):
        return False
    if not _equal_ci(metadata.get("case_citation"), filters.case_citation):
        return False

    candidate_date = metadata.get("decision_date") or metadata.get("effective_date")
    if filters.date_from or filters.date_to:
        if not candidate_date:
            return False
        try:
            normalized = date.fromisoformat(str(candidate_date)).isoformat()
        except ValueError:
            return False
        if filters.date_from and normalized < filters.date_from:
            return False
        if filters.date_to and normalized > filters.date_to:
            return False

    return True
