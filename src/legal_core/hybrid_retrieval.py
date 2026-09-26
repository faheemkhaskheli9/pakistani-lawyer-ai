"""Hybrid lexical + vector retrieval for legal text.

The lexical side uses a small dependency-free BM25 implementation so exact
legal terms, section numbers and citation strings can influence ranking even
when semantic embeddings are weak. Scores are normalized to [0, 1] before
weighted fusion, preserving the score contract used by the relevance gate.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .embeddings import get_embedder
from .retrieval import DEFAULT_INDEX_DIR, RetrievedChunk, Retriever, resolve_top_k
from .vector_store import VectorMatch, VectorStore

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
DEFAULT_VECTOR_WEIGHT = 0.65


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


@dataclass(frozen=True)
class LexicalMatch:
    id: str
    score: float
    document: str
    metadata: dict


class BM25Index:
    def __init__(self, entries: list[VectorMatch], *, k1: float = 1.5, b: float = 0.75):
        self.entries = entries
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(entry.document) for entry in entries]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._avgdl = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

        self._dfs: Counter[str] = Counter()
        for tokens in self._tokens:
            self._dfs.update(set(tokens))

    def search(self, query: str, *, top_k: int) -> list[LexicalMatch]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_tokens = tokenize(query)
        if not query_tokens or not self.entries:
            return []

        n_docs = len(self.entries)
        raw: list[tuple[int, float]] = []
        for index, tokens in enumerate(self._tokens):
            if not tokens:
                continue
            freqs = Counter(tokens)
            score = 0.0
            dl = self._lengths[index]
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                df = self._dfs[term]
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                norm = tf + self.k1 * (
                    1.0 - self.b + self.b * dl / (self._avgdl or 1.0)
                )
                score += idf * (tf * (self.k1 + 1.0)) / norm
            if score > 0:
                raw.append((index, score))

        raw.sort(key=lambda item: (-item[1], item[0]))
        raw = raw[:top_k]
        max_score = raw[0][1] if raw else 0.0
        return [
            LexicalMatch(
                id=self.entries[index].id,
                score=(score / max_score if max_score else 0.0),
                document=self.entries[index].document,
                metadata=dict(self.entries[index].metadata),
            )
            for index, score in raw
        ]


class HybridRetriever:
    """Fuse BM25 and vector rankings while returning normal RetrievedChunk objects."""

    def __init__(
        self,
        vector_retriever: Retriever,
        *,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        top_k: int | None = None,
    ):
        if not 0.0 <= vector_weight <= 1.0:
            raise ValueError("vector_weight must be between 0 and 1")
        self.vector_retriever = vector_retriever
        self.vector_store = vector_retriever.vector_store
        self.vector_weight = float(vector_weight)
        self.lexical_weight = 1.0 - self.vector_weight
        self.top_k = resolve_top_k(top_k if top_k is not None else vector_retriever.top_k)
        self.lexical_index = BM25Index(self.vector_store.entries())

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []
        k = resolve_top_k(top_k if top_k is not None else self.top_k, env={})
        if len(self.vector_store) == 0:
            return []

        candidate_k = min(len(self.vector_store), max(k * 3, k))
        vector_results = self.vector_retriever.retrieve(query, top_k=candidate_k)
        lexical_results = self.lexical_index.search(query, top_k=candidate_k)

        vector_by_id = {item.chunk_id: item for item in vector_results}
        lexical_by_id = {item.id: item for item in lexical_results}
        vector_max = max((max(item.score, 0.0) for item in vector_results), default=0.0)

        ordered_ids = list(dict.fromkeys(
            [item.chunk_id for item in vector_results] + [item.id for item in lexical_results]
        ))
        fused = []
        for stable_index, chunk_id in enumerate(ordered_ids):
            vector = vector_by_id.get(chunk_id)
            lexical = lexical_by_id.get(chunk_id)
            vector_score = max(vector.score, 0.0) / vector_max if vector and vector_max else 0.0
            lexical_score = lexical.score if lexical else 0.0
            score = self.vector_weight * vector_score + self.lexical_weight * lexical_score

            if vector is not None:
                text = vector.text
                source = vector.source
                section = vector.section
                chunk_index = vector.chunk_index
            else:
                assert lexical is not None
                text = lexical.document
                source = lexical.metadata.get("source")
                section = lexical.metadata.get("section")
                chunk_index = lexical.metadata.get("chunk_index")

            fused.append((score, stable_index, chunk_id, text, source, section, chunk_index))

        fused.sort(key=lambda row: (-row[0], row[1]))
        return [
            RetrievedChunk(
                rank=rank,
                chunk_id=chunk_id,
                score=min(max(float(score), 0.0), 1.0),
                text=text,
                source=source,
                section=section,
                chunk_index=chunk_index,
                metadata=(
                    dict(vector.metadata)
                    if vector is not None
                    else dict(lexical.metadata)
                ),
            )
            for rank, (score, _, chunk_id, text, source, section, chunk_index)
            in enumerate(fused[:k], start=1)
        ]


def build_hybrid_retriever(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    top_k: int | None = None,
    embedding_provider: str | None = None,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
) -> HybridRetriever:
    store = VectorStore(Path(index_dir) / "vectors")
    vector = Retriever(get_embedder(embedding_provider), store, top_k=top_k)
    return HybridRetriever(vector, vector_weight=vector_weight, top_k=top_k)
