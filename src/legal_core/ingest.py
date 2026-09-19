"""Corpus ingestion pipeline (issue #1, Phase 1) and CLI (issue #4, Phase 1).

Parses `.txt`/`.pdf` source documents, chunks them, tags each chunk with a
source citation, and upserts into a `ChunkStore`. A malformed source
document is skipped with a logged error — it never aborts the whole run.

Pattern: this portfolio's `rag/chunking-and-embedding-ingestion` knowledge-
base entry — chunk id is a deterministic hash of (source, chunk_index,
chunk_text), never a counter/timestamp, so re-running ingestion on an
unchanged corpus upserts the same ids (idempotent) while an edited chunk
gets a new id instead of silently overwriting old text under a stale key.

`python -m legal_core.ingest --corpus <dir>` runs this ingestion pipeline
and the Phase-1 embedding step (`legal_core.embedding_pipeline`) end to end
against a corpus directory (e.g. the checked-in seed corpus at
`data/corpus`), per this project's README §9. Both underlying stores
(`ChunkStore`, `VectorStore`) upsert by deterministic id, so re-running the
same command against an unchanged corpus is a no-op on the index rather
than growing it.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Iterable, List

from .chunk_store import ChunkRecord, ChunkStore
from .chunking import chunk_text
from .citation import extract_section_id
from .embedding_pipeline import embed_new_chunks
from .embeddings import get_embedder
from .parsing import SUPPORTED_EXTENSIONS, DocumentParseError, read_source_text
from .vector_store import VectorStore

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


def run_ingestion(
    corpus_dir: str | Path,
    index_dir: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    embedding_provider: str | None = None,
) -> tuple[int, int]:
    """Run ingestion + embedding end to end against `corpus_dir`.

    Persists a `ChunkStore` at `<index_dir>/chunks.json` and a `VectorStore`
    at `<index_dir>/vectors` (both upsert-by-id and atomic on write — see
    their own modules). Returns `(chunks_ingested, chunks_newly_embedded)`.

    `corpus_dir` is caller-supplied, explicit input (a CLI flag, never a
    silently-applied default), so a bad path is a hard failure here: this
    propagates `FileNotFoundError` from `iter_source_files` rather than
    swallowing it, per this project's "strict on explicit input" rule.

    Idempotent: chunk ids are a deterministic hash of (source, chunk_index,
    chunk_text) and both stores upsert by id, so re-running against an
    unchanged corpus re-touches the same records instead of duplicating
    them — `chunks_newly_embedded` is 0 on a repeat run with nothing new.
    """
    index_dir = Path(index_dir)
    chunk_store = ChunkStore(index_dir / "chunks.json")
    vector_store = VectorStore(index_dir / "vectors")

    pipeline = IngestionPipeline(chunk_store, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ingested = pipeline.ingest_directory(corpus_dir)

    embedder = get_embedder(embedding_provider)
    embedded = embed_new_chunks(chunk_store, embedder, vector_store)
    return ingested, embedded


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m legal_core.ingest",
        description=(
            "Ingest a directory of .txt/.pdf legal source documents: chunk "
            "it, tag each chunk with a source citation, and embed it into "
            "the local vector store. Safe to re-run against an unchanged "
            "corpus -- already-ingested/embedded chunks are upserted by id, "
            "never duplicated."
        ),
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Directory of .txt/.pdf source documents to ingest, e.g. data/corpus for the checked-in seed corpus.",
    )
    parser.add_argument(
        "--index-dir",
        default="data/index",
        help="Directory to persist the chunk store and vector store in (default: data/index).",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="Max characters per chunk (default: 800).")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Character overlap between chunks (default: 100).")
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Embedding provider name (default: $EMBEDDING_PROVIDER env var, or 'hashing' -- CPU-only, no model download).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: `python -m legal_core.ingest --corpus data/corpus`.

    Exits non-zero on a real ingestion failure (bad/missing corpus
    directory, malformed seed-corpus manifest, unknown embedding provider,
    ...) instead of exiting 0 having silently done nothing -- a per-file
    malformed document is still skip-and-logged (see `ingest_directory`),
    which is not a "real ingestion failure" in that sense.
    """
    args = _build_arg_parser().parse_args(argv)
    try:
        ingested, embedded = run_ingestion(
            corpus_dir=args.corpus,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_provider=args.embedding_provider,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure
        # here must become a non-zero exit for the caller (CI, a human, a
        # script) rather than an uncaught traceback or a silent success.
        logger.error("Ingestion failed: %s", exc)
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Ingestion complete: {ingested} chunk(s) processed, {embedded} newly embedded "
        f"(index: {Path(args.index_dir).resolve()})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    sys.exit(main())
