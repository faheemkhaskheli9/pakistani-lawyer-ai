"""Fixed-size character chunking with overlap.

Pattern: this portfolio's `rag/chunking-and-embedding-ingestion` knowledge-
base entry — whitespace-aware boundaries, and overlap is guarded so it can
never stall/move `start` backward (infinite-loop risk near `chunk_size`).
"""
from __future__ import annotations

import re
from typing import List


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """Split `text` into overlapping chunks of at most `chunk_size` chars.

    Boundaries snap to the nearest preceding whitespace when one exists
    within the chunk, so words aren't split mid-token.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        next_start = end - chunk_overlap
        # Guard against the whitespace-snapped `end` landing so close to
        # `start` that overlap would produce zero forward progress.
        start = next_start if next_start > start else end
    return chunks


_HEADING_RE = re.compile(r"^(?:Section|Article)\s+\d+[A-Z]?\s*[.:\-]", re.I | re.M)


def chunk_by_section(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """Split on statute headings so each chunk begins at a section boundary.

    Any preamble before the first heading becomes its own chunk, and a section
    longer than `chunk_size` is sub-chunked with `chunk_text`. Text without
    headings falls back to plain fixed-size chunking.
    """
    text = text.strip()
    starts = [m.start() for m in _HEADING_RE.finditer(text)]
    if not starts:
        return chunk_text(text, chunk_size, chunk_overlap)

    bounds = ([0] if starts[0] > 0 else []) + starts + [len(text)]
    chunks: List[str] = []
    for begin, end in zip(bounds, bounds[1:]):
        chunks.extend(chunk_text(text[begin:end], chunk_size, chunk_overlap))
    return chunks
