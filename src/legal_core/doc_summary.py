"""Extractive summary of an uploaded legal document (FIR, contract, notice).

No model is involved: key sentences are quoted verbatim with their character
offsets, so every line of the summary points back to the exact passage.  Dates,
money amounts, statutory sections and reported-case citations are pulled out
with regexes.  Output is a reading aid, not legal advice.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .citation_parser import find_citations
from .safeguards import LEGAL_INFORMATION_DISCLAIMER

MAX_CHARS = 200_000

_KIND_KEYWORDS = {
    "fir": ("first information report", "f.i.r", "fir no", "police station", "complainant", "ایف آئی آر"),
    "contract": ("agreement", "hereinafter", "party of the first part", "witnesseth", "whereas", "عقد", "معاہدہ"),
    "legal_notice": ("legal notice", "hereby called upon", "under instructions", "failing which", "نوٹس"),
}
_SENT_RE = re.compile(r"(?<=[.!?۔؟])\s+|\n{2,}")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})\b",
    re.I,
)
_AMOUNT_RE = re.compile(r"\b(?:Rs\.?|PKR)\s?[\d,]+(?:\.\d+)?(?:\s?(?:lakh|lac|crore|million))?", re.I)
_SECTION_RE = re.compile(
    r"\b(?:sections?|s\.|u/s)\s*(\d+[A-Z]?(?:\s*(?:,|and|/)\s*\d+[A-Z]?)*)\s*(PPC|CrPC|Cr\.P\.C|CPC|Contract Act|ATA|Limitation Act|Qanun-e-Shahadat)?",
    re.I,
)
_STOP = frozenset("the of and to in a is that for on by with as be this shall was are or at from it an".split())


@dataclass(frozen=True)
class SummaryPoint:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class DocumentSummary:
    kind: str
    characters: int
    key_points: tuple[SummaryPoint, ...]
    dates: tuple[str, ...]
    amounts: tuple[str, ...]
    sections: tuple[str, ...]
    case_citations: tuple[str, ...]
    disclaimer: str = LEGAL_INFORMATION_DISCLAIMER


def guess_kind(text: str) -> str:
    low = text.lower()
    scores = {k: sum(low.count(w) for w in ws) for k, ws in _KIND_KEYWORDS.items()}
    kind, score = max(scores.items(), key=lambda kv: kv[1])
    return kind if score else "unknown"


def _uniq(items) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(" ".join(item.split()), None)
    return tuple(seen)


def _sentences(text: str):
    pos = 0
    for part in _SENT_RE.split(text):
        idx = text.find(part, pos)
        if idx < 0 or not part.strip():
            continue
        pos = idx + len(part)
        stripped = part.strip()
        yield stripped, idx + (len(part) - len(part.lstrip()))
        

def summarize_document(text: str, *, max_points: int = 5) -> DocumentSummary:
    if not text or not text.strip():
        raise ValueError("document text must not be empty")
    if len(text) > MAX_CHARS:
        raise ValueError(f"document too large (max {MAX_CHARS} characters)")
    sents = [(s, a) for s, a in _sentences(text) if len(s) >= 25]
    freq = Counter(
        w for s, _ in sents for w in re.findall(r"\w+", s.lower()) if w not in _STOP and len(w) > 2
    )

    def score(item: tuple[str, int]) -> float:
        s, _ = item
        words = [w for w in re.findall(r"\w+", s.lower()) if w not in _STOP and len(w) > 2]
        base = sum(freq[w] for w in words) / (len(words) or 1)
        bonus = 2 * bool(_DATE_RE.search(s)) + 2 * bool(_AMOUNT_RE.search(s)) + 2 * bool(_SECTION_RE.search(s))
        return base + bonus

    top = sorted(sents, key=score, reverse=True)[:max_points]
    points = tuple(
        SummaryPoint(s, a, a + len(s)) for s, a in sorted(top, key=lambda item: item[1])
    )
    sections = _uniq(
        f"Section {m.group(1)}" + (f" {m.group(2)}" if m.group(2) else "")
        for m in _SECTION_RE.finditer(text)
    )
    return DocumentSummary(
        kind=guess_kind(text),
        characters=len(text),
        key_points=points,
        dates=_uniq(m.group(0) for m in _DATE_RE.finditer(text)),
        amounts=_uniq(m.group(0) for m in _AMOUNT_RE.finditer(text)),
        sections=sections,
        case_citations=_uniq(c.normalized for c in find_citations(text)),
    )
