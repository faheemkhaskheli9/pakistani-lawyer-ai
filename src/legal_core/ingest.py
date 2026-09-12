"""Corpus ingestion pipeline (issue #1, Phase 1).

Parses `.txt`/`.pdf` source documents, chunks them, tags each chunk with a
source citation, and upserts into a `ChunkStore`. A malformed source
document is skipped with a logged error — it never aborts the whole run.

Pattern: this portfolio's `rag/chunking-and-embedding-ingestion` knowledge-
base entry — chunk id is a deterministic hash of (source, chunk_index,
chunk_text), never a counter/timestamp, so re-running ingestion on an
unchanged corpus upserts the same ids (idempotent) while an edited chunk
gets a new id instead of silently overwriting old text under a stale key.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable, List

from .chunk_store import ChunkRecord, ChunkStore
from .chunking import chunk_text
from .citation import extract_section_id
from .parsing import SUPPORTED_EXTENSIONS, DocumentParseError, read_source_text

logger = logging.getLogger(__name__)


def _chunk_id(source: str, chunk_index: int, chunk: str) -> str:
    digest = hashlib.sha256(f"{source}::{chunk_index}::{chunk}".encode("utf-8")).hexdigest()
    return digest


def iter_source_files(source_dir: str | Path, extensions: Iterable[str] = SUPPORTED_EXTENSIONS) -> List[Path]:
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    extensions = set(extensions)
    return sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


class IngestionPipeline:
    def __init__(self, store: ChunkStore, chunk_size: int = 800, chunk_overlap: int = 100):
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, path: Path) -> int:
        """Ingest one file; returns the number of new/updated chunks.

        Raises `DocumentParseError` on a malformed document — callers that
        want skip-and-continue behavior over a whole directory should use
        `ingest_directory`, which catches this per file.
        """
        text = read_source_text(path)
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            logger.info("No content chunks extracted from %s", path)
            return 0

        records = [
            ChunkRecord(
                id=_chunk_id(str(path), i, chunk),
                source=str(path),
                section=extract_section_id(chunk, i),
                chunk_index=i,
                text=chunk,
                metadata={"title": path.stem},
            )
            for i, chunk in enumerate(chunks)
        ]
        self.store.upsert(records)
        logger.info("Ingested %s: %d chunks", path, len(records))
        return len(records)

    def ingest_directory(self, source_dir: str | Path) -> int:
        """Ingest every supported file under `source_dir`.

        A malformed source document is logged and skipped; ingestion
        continues with the remaining files rather than aborting the run.
        """
        total = 0
        for path in iter_source_files(source_dir):
            try:
                total += self.ingest_file(path)
            except DocumentParseError as exc:
                logger.error("Skipping malformed source document %s: %s", path, exc)
                continue
        logger.info("Ingestion run complete: %d chunks written from %s", total, source_dir)
        return total
