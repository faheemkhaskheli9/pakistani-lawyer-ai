"""Embedding provider interface (issue #2, Phase 1).

Pattern: this portfolio's `rag/chunking-and-embedding-ingestion`
knowledge-base entry -- the embedding provider is an interface, not a
hard dependency, so the pipeline is CPU-only and testable with no model
download/API key by default. `HashingEmbedder` is that dependency-free
default (a deterministic hashing/bag-of-words embedding, L2-normalized);
`SentenceTransformerEmbedder` is the real implementation the acceptance
criteria ask for, imported lazily (mirrors `agents_core.backend`'s
`OpenAIBackend` in the OpenAI-Agents-App sibling repo) so `sentence-
transformers` and its model download are only needed when that provider
is actually selected.
"""
from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Common interface every embedding backend implements."""

    #: Length of every vector this provider returns.
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each of `texts`, returning one `dimension`-length vector
        (L2-normalized) per input, in the same order."""
        raise NotImplementedError


class HashingEmbedder(EmbeddingProvider):
    """Deterministic, offline bag-of-words-into-N-buckets embedder.

    No model, no network, no API key -- the same text always hashes to the
    same vector, so it's reproducible across runs/tests while still
    reflecting real lexical overlap (shared words hash into the same
    buckets), which is enough to prove the ingestion/retrieval pipeline
    works CPU-only before a real model is wired in.
    """

    def __init__(self, dimension: int = 256):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = text.lower().split()
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Real embedding provider backed by `sentence-transformers`.

    The package (and its model download) is imported lazily inside
    `__init__`, not at module import time, so selecting the offline
    `HashingEmbedder` default never requires it to be installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - only hit when selected
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'sentence-transformers' "
                "package: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


#: Registry of provider name -> implementation class (extend by adding an
#: entry here, not by branching in `get_embedder`).
_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "hashing": HashingEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
}


def get_embedder(name: str | None = None) -> EmbeddingProvider:
    """Return the configured embedding provider.

    `name` overrides the `EMBEDDING_PROVIDER` env var (default: "hashing").
    Swapping providers is config only, mirroring `agents_core.get_backend`
    in the OpenAI-Agents-App sibling repo.
    """
    key = (name or os.environ.get("EMBEDDING_PROVIDER") or "hashing").strip().lower()
    try:
        provider_cls = _REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown EMBEDDING_PROVIDER '{key}' (known: {known})") from exc
    return provider_cls()
