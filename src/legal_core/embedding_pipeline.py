"""Embed ingested chunks into the local vector store (issue #2, Phase 1).

Ties `ChunkStore` (Phase 1 ingestion output) to an `EmbeddingProvider` and
a `VectorStore`. A chunk already present in the vector store is skipped
without calling the embedder again -- a no-op, not a duplicate -- so
re-running embedding after re-ingesting an unchanged corpus does no
redundant work (idempotent, per this issue's acceptance criteria).
"""
from __future__ import annotations

from .chunk_store import ChunkStore
from .embeddings import EmbeddingProvider
from .vector_store import VectorStore


def embed_new_chunks(store: ChunkStore, embedder: EmbeddingProvider, vector_store: VectorStore) -> int:
    """Embed every chunk in `store` not yet in `vector_store`.

    Returns the number of chunks actually embedded (0 if the vector store
    was already up to date with `store`).
    """
    pending = [record for record in store.all() if record.id not in vector_store]
    if not pending:
        return 0

    embeddings = embedder.embed([record.text for record in pending])
    for record, embedding in zip(pending, embeddings):
        vector_store.upsert(
            record.id,
            embedding,
            record.text,
            metadata={
                **record.metadata,
                "source": record.source,
                "section": record.section,
                "chunk_index": record.chunk_index,
            },
        )
    return len(pending)
