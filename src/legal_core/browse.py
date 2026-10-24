"""Read-only corpus browsing: list ingested documents and their sections."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .chunk_store import ChunkStore
from .retrieval import DEFAULT_INDEX_DIR


@dataclass(frozen=True)
class CorpusDocument:
    source: str
    chunks: int
    sections: tuple[str, ...]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusSection:
    section: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class CorpusDocumentDetail:
    source: str
    metadata: dict
    sections: tuple[CorpusSection, ...]


class CorpusBrowseService:
    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)

    def _records(self):
        return ChunkStore(self.index_dir / "chunks.json").all()

    @staticmethod
    def _name(source: str) -> str:
        return PurePosixPath(source.replace("\\", "/")).name

    def list_documents(self) -> tuple[CorpusDocument, ...]:
        grouped: dict[str, list] = {}
        for record in self._records():
            grouped.setdefault(self._name(record.source), []).append(record)
        return tuple(
            CorpusDocument(
                source=name,
                chunks=len(records),
                sections=tuple(dict.fromkeys(r.section for r in records if r.section)),
                metadata=dict(records[0].metadata),
            )
            for name, records in sorted(grouped.items())
        )

    def get_document(self, source: str) -> CorpusDocumentDetail | None:
        name = self._name(source)
        records = sorted(
            (r for r in self._records() if self._name(r.source) == name),
            key=lambda r: r.chunk_index,
        )
        if not records:
            return None
        return CorpusDocumentDetail(
            source=name,
            metadata=dict(records[0].metadata),
            sections=tuple(
                CorpusSection(section=r.section, chunk_index=r.chunk_index, text=r.text)
                for r in records
            ),
        )
