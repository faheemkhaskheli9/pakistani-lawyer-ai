"""Local, dependency-free vector store (issue #2, Phase 1).

Pattern: this portfolio's `rag/chunking-and-embedding-ingestion`
knowledge-base entry -- a vector store is an interface
(`upsert`/`query`), and a dependency-free local implementation (here:
numpy cosine similarity, persisted as one `.npz` + one JSON metadata
file) stands in for a real service (FAISS/Chroma) so retrieval has no
external service dependency, matching this issue's acceptance criteria.
Upsert is keyed by the same deterministic chunk id `legal_core.ingest`
already computes, so re-embedding an unchanged chunk overwrites the same
row instead of creating a duplicate.

Persisted atomically: both files are written to a temp path in the same
directory and `os.replace()`d onto the target only after both succeed, so
an interrupted save (mid-write crash, disk full) never leaves a
half-written store on disk for the next reload to trust.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class DimensionMismatch(ValueError):
    """Raised when an upserted embedding's length doesn't match the store's
    established dimension -- catching a caller accidentally mixing
    embedding providers in one store instead of silently corrupting
    similarity search."""


@dataclass(frozen=True)
class VectorMatch:
    id: str
    score: float
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """Upsert-by-id local vector store with cosine-similarity search."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None  # shape (n, dimension)
        self._documents: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self.dimension: int | None = None
        self._load()

    def _vectors_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".npz")

    def _meta_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".json")

    def _load(self) -> None:
        vec_path, meta_path = self._vectors_path(), self._meta_path()
        if not vec_path.exists() or not meta_path.exists():
            return
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        self._ids = meta["ids"]
        self._documents = meta["documents"]
        self._metadata = meta["metadata"]
        with np.load(vec_path) as npz:
            self._vectors = npz["vectors"]
        if self._vectors is not None and len(self._vectors) > 0:
            self.dimension = self._vectors.shape[1]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        vec_path, meta_path = self._vectors_path(), self._meta_path()
        vec_tmp = vec_path.with_suffix(vec_path.suffix + ".tmp")
        meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
        vectors = self._vectors if self._vectors is not None else np.zeros((0, 0))
        try:
            # Pass an open file handle (not a path string) so numpy never
            # guesses/appends a ".npz" suffix onto our own ".tmp" name.
            with vec_tmp.open("wb") as fh:
                np.savez(fh, vectors=vectors)
            with meta_tmp.open("w", encoding="utf-8") as fh:
                json.dump(
                    {"ids": self._ids, "documents": self._documents, "metadata": self._metadata},
                    fh,
                )
            os.replace(vec_tmp, vec_path)
            os.replace(meta_tmp, meta_path)
        except Exception:
            vec_tmp.unlink(missing_ok=True)
            meta_tmp.unlink(missing_ok=True)
            raise

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._documents

    def upsert(self, id_: str, embedding: list[float], document: str, metadata: dict[str, Any] | None = None) -> None:
        if self.dimension is None:
            self.dimension = len(embedding)
        elif len(embedding) != self.dimension:
            raise DimensionMismatch(
                f"Embedding for {id_!r} has dimension {len(embedding)}, "
                f"store expects {self.dimension}"
            )

        metadata = metadata or {}
        row = np.array(embedding, dtype="float32")
        if id_ in self._documents:
            index = self._ids.index(id_)
            self._vectors[index] = row
        else:
            self._ids.append(id_)
            if self._vectors is None or len(self._vectors) == 0:
                self._vectors = row.reshape(1, -1)
            else:
                self._vectors = np.vstack([self._vectors, row])
        self._documents[id_] = document
        self._metadata[id_] = metadata
        self._save()

    def query(self, embedding: list[float], top_k: int = 5) -> list[VectorMatch]:
        if not self._ids:
            return []
        query_vec = np.array(embedding, dtype="float32")
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        vectors = self._vectors
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1e-12  # avoid dividing a zero-vector row by zero
        scores = (vectors @ query_vec) / (norms * query_norm)

        top_k = max(0, min(top_k, len(self._ids)))
        top_indices = np.argsort(-scores)[:top_k]
        return [
            VectorMatch(
                id=self._ids[i],
                score=float(scores[i]),
                document=self._documents[self._ids[i]],
                metadata=self._metadata[self._ids[i]],
            )
            for i in top_indices
        ]
