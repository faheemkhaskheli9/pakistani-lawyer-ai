"""Seed corpus of small, original sample statute-style text (issue #3, Phase 1).

The project ships with a tiny, checked-in corpus under `data/corpus/` so
`legal_core`'s ingestion pipeline (`legal_core.ingest`) is runnable and
testable end-to-end with no external download and no network access.

Every seed document is **original sample text written for this project** in
a generic, clearly-fictional statute style ("Sample Civil Code", "Sample
Criminal Procedure Code", "Sample Evidence Act") -- not a verbatim
reproduction of, or derived from, any specific real Pakistani statute -- so
there is no copyright question about what's checked into git, and no risk
of it being mistaken for real legal authority. `sources.json` alongside the
`.txt` files records, per document, the source basis under which it's
included here (this is what issue #3's "each seed document records its
source/license basis" acceptance criterion asks for).

The seed corpus is ingested through the exact same `IngestionPipeline` as
any other corpus -- there is no seed-only ingestion code path.
"""
from __future__ import annotations

import json
from pathlib import Path

from .chunk_store import ChunkStore
from .ingest import IngestionPipeline

#: Directory holding the checked-in seed corpus (.txt files + sources.json).
SEED_CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "corpus"

#: Filename of the manifest recording each seed document's source/license basis.
SOURCES_MANIFEST_NAME = "sources.json"

_REQUIRED_MANIFEST_FIELDS = ("title", "source_url", "basis")


def load_sources_manifest(corpus_dir: str | Path = SEED_CORPUS_DIR) -> dict[str, dict[str, str | None]]:
    """Load and validate the per-document source/license manifest for `corpus_dir`.

    This is a small, checked-in file the project itself controls (not
    external/user input), so we're strict about it: a missing or malformed
    manifest is a hard error rather than a silent empty result, and every
    `.txt` seed document must have a corresponding entry with all of
    `_REQUIRED_MANIFEST_FIELDS` present -- a document with no recorded
    source/license basis would silently violate this issue's acceptance
    criterion, so we fail loudly instead.
    """
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / SOURCES_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Seed corpus manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"Seed corpus manifest is empty or malformed: {manifest_path}")

    seed_files = {p.name for p in corpus_dir.glob("*.txt")}
    missing_entries = seed_files - manifest.keys()
    if missing_entries:
        raise ValueError(
            f"Seed document(s) missing from {manifest_path}: {sorted(missing_entries)}"
        )

    for filename, entry in manifest.items():
        if not isinstance(entry, dict) or any(field not in entry for field in _REQUIRED_MANIFEST_FIELDS):
            raise ValueError(
                f"Manifest entry for {filename!r} is missing one of "
                f"{_REQUIRED_MANIFEST_FIELDS} in {manifest_path}"
            )
        if not entry.get("basis"):
            raise ValueError(f"Manifest entry for {filename!r} has no recorded source/license basis")

    return manifest


def load_seed_corpus(
    store: ChunkStore,
    corpus_dir: str | Path = SEED_CORPUS_DIR,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> int:
    """Ingest the checked-in seed corpus into `store` and return the chunk count.

    Validates the source/license manifest first (see `load_sources_manifest`)
    so a corpus that's missing its licensing record fails before anything is
    written, then runs the same `IngestionPipeline.ingest_directory` any
    other corpus directory would use.
    """
    load_sources_manifest(corpus_dir)
    pipeline = IngestionPipeline(store, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return pipeline.ingest_directory(corpus_dir)
