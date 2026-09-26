"""Top-k retrieval over the local vector store (issue #5, Phase 2).

`Retriever.retrieve(query)` embeds a free-text query with the same
`EmbeddingProvider` the index was built with and returns the `k` most similar
chunks, best first, each carrying the source/section it was ingested with so
the answer-generation step can cite it.

An index that was never built, or is empty, yields `[]` -- "no grounded
context", which the caller must handle -- not an exception. Caller mistakes
(a non-positive `k`, querying with a different embedding provider than the
index was built with) still raise: returning nothing for those would be
indistinguishable from "no relevant source found".

    python -m legal_core.retrieval "What is the limitation period?" --top-k 3

Runs CPU-only with the default hashing embedder; no network or paid API.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import EmbeddingProvider, get_embedder
from .vector_store import VectorStore

DEFAULT_TOP_K = 5
TOP_K_ENV = "RETRIEVAL_TOP_K"
DEFAULT_INDEX_DIR = "data/index"


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int            # 1 = most similar
    chunk_id: str
    score: float         # cosine similarity
    text: str
    source: str | None   # source document the chunk was ingested from
    section: str | None  # section/paragraph citation within that source
    chunk_index: int | None
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        parts = [part for part in (self.source, self.section) if part]
        return ", ".join(parts) if parts else self.chunk_id


def _validate_top_k(value: object, origin: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{origin} must be a positive integer, got {value!r}")
    return value


def resolve_top_k(top_k: int | None = None, env: dict | None = None) -> int:
    """`top_k` if given, else `$RETRIEVAL_TOP_K`, else `DEFAULT_TOP_K`.

    An explicitly supplied value -- argument or env var -- that is not a
    positive integer is an error, never a silent fall back to the default.
    """
    if top_k is not None:
        return _validate_top_k(top_k, "top_k")
    env = os.environ if env is None else env
    raw = env.get(TOP_K_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_TOP_K
    try:
        parsed = int(raw.strip())
    except ValueError:
        raise ValueError(f"${TOP_K_ENV} must be a positive integer, got {raw!r}") from None
    return _validate_top_k(parsed, f"${TOP_K_ENV}")


class Retriever:
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore, top_k: int | None = None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = resolve_top_k(top_k)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Chunks ranked by descending similarity to `query`, at most `top_k`
        (default: this retriever's configured `top_k`)."""
        k = self.top_k if top_k is None else _validate_top_k(top_k, "top_k")
        if not query or not query.strip():
            return []
        if len(self.vector_store) == 0:
            return []  # empty or never-built index: nothing to embed the query for

        [embedding] = self.embedder.embed([query.strip()])
        matches = self.vector_store.query(embedding, top_k=k)
        return [
            RetrievedChunk(
                rank=rank,
                chunk_id=match.id,
                score=match.score,
                text=match.document,
                source=match.metadata.get("source"),
                section=match.metadata.get("section"),
                chunk_index=match.metadata.get("chunk_index"),
                metadata=dict(match.metadata),
            )
            for rank, match in enumerate(matches, start=1)
        ]


def build_retriever(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    top_k: int | None = None,
    embedding_provider: str | None = None,
) -> Retriever:
    """Retriever over the index `python -m legal_core.ingest` writes to
    `<index_dir>/vectors`. `embedding_provider` must match the one used at
    ingestion time (default: `$EMBEDDING_PROVIDER`, else "hashing")."""
    return Retriever(
        get_embedder(embedding_provider),
        VectorStore(Path(index_dir) / "vectors"),
        top_k=top_k,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m legal_core.retrieval",
        description="Print the top-k most similar corpus chunks for a query, with their citations.",
    )
    parser.add_argument("query", help="Free-text question or search phrase.")
    parser.add_argument(
        "--index-dir", default=DEFAULT_INDEX_DIR,
        help=f"Index directory written by legal_core.ingest (default: {DEFAULT_INDEX_DIR}).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help=f"Number of chunks to return (default: ${TOP_K_ENV} env var, or {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--embedding-provider", default=None,
        help="Must match the provider used at ingestion (default: $EMBEDDING_PROVIDER, or 'hashing').",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        retriever = build_retriever(
            args.index_dir, top_k=args.top_k, embedding_provider=args.embedding_provider
        )
        results = retriever.retrieve(args.query)
    except Exception as exc:  # noqa: BLE001 - any failure must become a non-zero exit, not a traceback
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1

    if not results:
        if len(retriever.vector_store) == 0:
            print(
                f"No results: the index at {Path(args.index_dir).resolve()} is empty. "
                "Run `python -m legal_core.ingest --corpus data/corpus` first."
            )
        else:
            print("No results.")
        return 0

    for chunk in results:
        preview = " ".join(chunk.text.split())
        if len(preview) > 160:
            preview = preview[:157] + "..."
        print(f"{chunk.rank}. [{chunk.score:.3f}] {chunk.citation}\n   {preview}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    sys.exit(main())
