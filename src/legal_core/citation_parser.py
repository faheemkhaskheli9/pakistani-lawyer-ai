"""Parser for common Pakistani reported-case citation formats."""
from __future__ import annotations

import re
from dataclasses import dataclass

_REPORTERS = ("PLD", "SCMR", "CLC", "MLD", "YLR")
_PLD_COURT = r"(?:SC|FSC|Lahore|Karachi|Peshawar|Quetta|Islamabad)"

_PLD_RE = re.compile(
    rf"\bPLD\s+(?P<year>\d{{4}})\s+(?P<court>{_PLD_COURT})\s+(?P<page>\d+)\b",
    re.I,
)
_STANDARD_RE = re.compile(
    r"\b(?P<year>\d{4})\s+(?P<reporter>SCMR|CLC|MLD|YLR)\s+(?P<page>\d+)\b",
    re.I,
)


@dataclass(frozen=True)
class ParsedCitation:
    reporter: str
    year: int
    page: int
    court: str | None
    normalized: str
    raw: str


def _normalize_court(value: str) -> str:
    upper = value.upper()
    if upper in {"SC", "FSC"}:
        return upper
    return value.title()


def parse_citation(text: str) -> ParsedCitation | None:
    """Parse one complete citation string, returning None when unsupported."""
    if not text or not text.strip():
        return None
    candidate = " ".join(text.strip().split())

    match = _PLD_RE.fullmatch(candidate)
    if match:
        year = int(match.group("year"))
        page = int(match.group("page"))
        court = _normalize_court(match.group("court"))
        return ParsedCitation(
            reporter="PLD",
            year=year,
            page=page,
            court=court,
            normalized=f"PLD {year} {court} {page}",
            raw=candidate,
        )

    match = _STANDARD_RE.fullmatch(candidate)
    if match:
        reporter = match.group("reporter").upper()
        year = int(match.group("year"))
        page = int(match.group("page"))
        return ParsedCitation(
            reporter=reporter,
            year=year,
            page=page,
            court=None,
            normalized=f"{year} {reporter} {page}",
            raw=candidate,
        )

    return None


def find_citations(text: str) -> tuple[ParsedCitation, ...]:
    """Find supported citations embedded in arbitrary text, in source order."""
    if not text:
        return ()

    matches: list[tuple[int, ParsedCitation]] = []
    for pattern in (_PLD_RE, _STANDARD_RE):
        for match in pattern.finditer(text):
            parsed = parse_citation(match.group(0))
            if parsed is not None:
                matches.append((match.start(), parsed))

    matches.sort(key=lambda item: item[0])
    seen: set[tuple[str, int, str | None, int]] = set()
    ordered: list[ParsedCitation] = []
    for _, parsed in matches:
        key = (parsed.reporter, parsed.year, parsed.court, parsed.page)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(parsed)
    return tuple(ordered)


def contains_supported_citation(text: str) -> bool:
    return bool(find_citations(text))
