"""Standalone "verify this citation" tool: check a citation string against the corpus."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .chunk_store import ChunkStore
from .citation_parser import parse_citation
from .retrieval import DEFAULT_INDEX_DIR

_STATUTE_RE = re.compile(
    r"^(?:section|sec\.?|s\.)\s*(?P<num>\d+[A-Za-z]?)\s*(?:of\s+(?:the\s+)?)?(?P<act>.*)$", re.I
)
_STOP = {"act", "the", "of", "code", "ordinance", "pakistan", "penal", "criminal"}


@dataclass(frozen=True)
class CitationMatch:
    source: str
    section: str


@dataclass(frozen=True)
class CitationVerification:
    citation: str
    kind: str  # "reported_case" | "statute_section" | "unrecognised"
    format_valid: bool
    found_in_corpus: bool
    normalized: str | None = None
    matches: tuple[CitationMatch, ...] = field(default_factory=tuple)
    note: str = ""


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def _name(source: str) -> str:
    return PurePosixPath(source.replace("\\", "/")).name


class CitationVerifier:
    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)

    def verify(self, citation: str) -> CitationVerification:
        text = " ".join((citation or "").split())
        if not text:
            raise ValueError("citation must not be empty")
        records = ChunkStore(self.index_dir / "chunks.json").all()

        parsed = parse_citation(text)
        if parsed is not None:
            needle = _squash(parsed.normalized)
            hits = tuple(
                CitationMatch(_name(r.source), r.section)
                for r in records
                if needle in _squash(r.text)
            )
            return CitationVerification(
                text, "reported_case", True, bool(hits), parsed.normalized, hits,
                "" if hits else "Citation is well-formed but not found in the loaded corpus; "
                "this does not prove it is fake -- check the law report.",
            )

        m = _STATUTE_RE.match(text)
        if m:
            num = m.group("num").lower()
            act_words = [w for w in re.findall(r"[a-z]+", m.group("act").lower()) if w not in _STOP]
            sec_re = re.compile(rf"\b(?:section|sec\.?|s\.)\s*{re.escape(num)}\b", re.I)
            hits = tuple(
                CitationMatch(_name(r.source), r.section)
                for r in records
                if sec_re.search(r.section or "")
                and all(w in _squash(_name(r.source) + " " + r.text) for w in act_words)
            )
            return CitationVerification(
                text, "statute_section", True, bool(hits), f"Section {m.group('num')}", hits,
                "" if hits else "Section not found in the loaded corpus.",
            )

        return CitationVerification(
            text, "unrecognised", False, False, None, (),
            "Unsupported format. Supported: PLD/SCMR/CLC/MLD/YLR reports and 'Section N <Act>'.",
        )
