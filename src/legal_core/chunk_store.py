"""Local, dependency-free store for ingested chunks.

Chunks land here before embedding (Phase 2 adds the vector store on top).
Persists as one JSON file, written atomically (temp file + rename) so an
interrupted write never leaves a truncated/corrupt store on disk.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ChunkRecord:
    id: str
    source: str
    section: str
    chunk_index: int
    text: str
    metadata: Dict = field(default_factory=dict)


class ChunkStore:
    """Upsert-by-id store — re-ingesting the same id never creates a
    duplicate (see issue #1's idempotency criterion)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: Dict[str, ChunkRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for item in raw:
            record = ChunkRecord(**item)
            self._records[record.id] = record

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump([asdict(r) for r in self._records.values()], fh, indent=2)
            tmp_path.replace(self.path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def upsert(self, records: List[ChunkRecord]) -> None:
        for record in records:
            self._records[record.id] = record
        self._save()

    def __len__(self) -> int:
        return len(self._records)

    def get_ids(self) -> set:
        return set(self._records.keys())

    def all(self) -> List[ChunkRecord]:
        return list(self._records.values())
