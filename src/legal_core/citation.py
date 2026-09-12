"""Per-chunk citation identifiers.

Statute/judgment text conventionally marks its own structure ("Section 5",
"Article 199", "Para 12"). We look for the first such marker inside a chunk
and use it as the citation; if none is present, we fall back to a stable
positional identifier so every chunk still has *something* to cite, even
plain prose with no numbered structure.
"""
from __future__ import annotations

import re

_SECTION_MARKERS = re.compile(
    r"\b(Section|Article|Chapter|Para(?:graph)?|Clause)\.?\s+(\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)


def extract_section_id(chunk: str, chunk_index: int) -> str:
    """Return a human-readable section/paragraph identifier for `chunk`.

    Prefers an explicit in-text marker (e.g. "Section 5", "Article 199");
    falls back to `para-<chunk_index + 1>` (1-based, so it reads naturally
    next to a citation) when the chunk carries no such marker.
    """
    match = _SECTION_MARKERS.search(chunk)
    if match:
        label, number = match.groups()
        return f"{label.title()} {number}"
    return f"para-{chunk_index + 1}"
