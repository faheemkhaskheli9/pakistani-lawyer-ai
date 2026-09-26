"""Statute hierarchy parsing for chunk-level legal context."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_PART_RE = re.compile(r"^\s*PART\s+([A-Z0-9IVXLC]+)\b(?:\s*[-—.:]?\s*(.*))?$", re.I)
_CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+([A-Z0-9IVXLC]+)\b(?:\s*[-—.:]?\s*(.*))?$", re.I)
_SECTION_RE = re.compile(r"^\s*(?:SECTION\s+)?(\d+[A-Z]?)\s*[.:-]\s*(.*)$", re.I)
_SUBSECTION_RE = re.compile(r"^\s*\((\d+[A-Z]?)\)\s+(.*)$", re.I)
_CLAUSE_RE = re.compile(r"^\s*\(([a-z]|[ivxlcdm]+)\)\s+(.*)$", re.I)
_EXPLANATION_RE = re.compile(r"^\s*EXPLANATION(?:\s+[IVXLC0-9]+)?\s*[.—:-]?\s*(.*)$", re.I)
_SCHEDULE_RE = re.compile(r"^\s*(?:(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)\s+)?SCHEDULE\b(?:\s*[-—.:]?\s*(.*))?$", re.I)


@dataclass(frozen=True)
class StatuteStructure:
    part: str | None = None
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    clause: str | None = None
    explanation: str | None = None
    schedule: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value}


class StatuteStructureTracker:
    """Track the latest hierarchy labels while chunks are processed in order."""

    def __init__(self):
        self.part = None
        self.chapter = None
        self.section = None
        self.subsection = None
        self.clause = None
        self.explanation = None
        self.schedule = None

    def update(self, text: str) -> StatuteStructure:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            match = _PART_RE.match(line)
            if match:
                self.part = f"Part {match.group(1).upper()}"
                self.chapter = self.section = self.subsection = self.clause = None
                self.explanation = self.schedule = None
                continue

            match = _CHAPTER_RE.match(line)
            if match:
                self.chapter = f"Chapter {match.group(1).upper()}"
                self.section = self.subsection = self.clause = None
                self.explanation = self.schedule = None
                continue

            match = _SCHEDULE_RE.match(line)
            if match:
                ordinal = match.group(1)
                self.schedule = (
                    f"{ordinal.title()} Schedule" if ordinal else "Schedule"
                )
                self.section = self.subsection = self.clause = None
                self.explanation = None
                continue

            match = _SECTION_RE.match(line)
            if match:
                self.section = f"Section {match.group(1)}"
                self.subsection = self.clause = None
                self.explanation = None
                continue

            match = _SUBSECTION_RE.match(line)
            if match:
                self.subsection = f"Subsection ({match.group(1)})"
                self.clause = None
                self.explanation = None
                continue

            match = _CLAUSE_RE.match(line)
            if match:
                self.clause = f"Clause ({match.group(1)})"
                self.explanation = None
                continue

            if _EXPLANATION_RE.match(line):
                self.explanation = "Explanation"

        return StatuteStructure(
            part=self.part,
            chapter=self.chapter,
            section=self.section,
            subsection=self.subsection,
            clause=self.clause,
            explanation=self.explanation,
            schedule=self.schedule,
        )


def extract_statute_structure(text: str) -> StatuteStructure:
    return StatuteStructureTracker().update(text)
