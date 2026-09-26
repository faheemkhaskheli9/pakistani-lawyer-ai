"""Deterministic legal-query intent and jurisdiction analysis."""
from __future__ import annotations

import re
from dataclasses import dataclass

INTENTS = {
    "statute_lookup",
    "case_search",
    "citation_lookup",
    "legal_question",
    "document_question",
    "legal_definition",
    "research_request",
}

_JURISDICTION_PATTERNS = (
    ("Punjab", re.compile(r"\bpunjab\b", re.I)),
    ("Sindh", re.compile(r"\bsindh\b", re.I)),
    ("Khyber Pakhtunkhwa", re.compile(r"\b(?:khyber pakhtunkhwa|kpk|kp)\b", re.I)),
    ("Balochistan", re.compile(r"\b(?:balochistan|baluchistan)\b", re.I)),
    ("Islamabad Capital Territory", re.compile(r"\b(?:islamabad|ict)\b", re.I)),
    ("Federal", re.compile(r"\b(?:federal|pakistan-wide|national law|constitution of pakistan)\b", re.I)),
)

_CITATION_RE = re.compile(
    r"\b(?:PLD\s+\d{4}\s+(?:SC|Lahore|Karachi|Peshawar|Quetta|Islamabad|FSC)\s+\d+"
    r"|\d{4}\s+(?:SCMR|CLC|MLD|YLR)\s+\d+)\b",
    re.I,
)


@dataclass(frozen=True)
class QueryAnalysis:
    intent: str
    jurisdiction: str | None
    normalized_query: str


def detect_jurisdiction(query: str) -> str | None:
    matches = [name for name, pattern in _JURISDICTION_PATTERNS if pattern.search(query)]
    if not matches:
        return None
    # A query mentioning multiple jurisdictions is deliberately not collapsed
    # into one, because choosing one would silently narrow legal research.
    if len(matches) > 1:
        return None
    return matches[0]


def classify_intent(query: str) -> str:
    text = " ".join(query.strip().split())
    lower = text.lower()

    if _CITATION_RE.search(text) or any(
        phrase in lower for phrase in ("citation ", "reported as", "find pl", "scmr", "clc ", "mld ", "ylr ")
    ):
        return "citation_lookup"

    if any(
        phrase in lower
        for phrase in (
            "this document",
            "uploaded document",
            "this contract",
            "this agreement",
            "this pdf",
            "this petition",
            "this notice",
        )
    ):
        return "document_question"

    if any(
        phrase in lower
        for phrase in (
            "define ",
            "definition of",
            "what does ",
            "meaning of",
            "what is meant by",
        )
    ):
        return "legal_definition"

    if any(
        phrase in lower
        for phrase in (
            "case law",
            "judgment",
            "judgement",
            "precedent",
            "decided case",
            "supreme court case",
            "high court case",
        )
    ):
        return "case_search"

    if any(
        phrase in lower
        for phrase in (
            "section ",
            "article ",
            "statute",
            "act ",
            "ordinance",
            "rules ",
            "regulation",
            "constitution",
        )
    ):
        return "statute_lookup"

    if any(
        phrase in lower
        for phrase in (
            "research memo",
            "research this",
            "authorities on",
            "find authorities",
            "compare authorities",
            "legal research",
        )
    ):
        return "research_request"

    return "legal_question"


def analyze_query(query: str) -> QueryAnalysis:
    if not query or not query.strip():
        raise ValueError("query must not be blank")
    normalized = " ".join(query.strip().split())
    return QueryAnalysis(
        intent=classify_intent(normalized),
        jurisdiction=detect_jurisdiction(normalized),
        normalized_query=normalized,
    )
