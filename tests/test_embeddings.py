"""Tests for issue #2's embedding provider interface. No network access or
model download is used here -- SentenceTransformerEmbedder is exercised
only up to the missing-package error path."""

from __future__ import annotations

import math

import pytest

from legal_core.embeddings import HashingEmbedder, SentenceTransformerEmbedder, get_embedder


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dimension=64)
    a = embedder.embed(["the quick brown fox"])[0]
    b = embedder.embed(["the quick brown fox"])[0]
    assert a == b


def test_hashing_embedder_differs_for_different_text():
    embedder = HashingEmbedder(dimension=64)
    a = embedder.embed(["contract law"])[0]
    b = embedder.embed(["criminal procedure"])[0]
    assert a != b


def test_hashing_embedder_vectors_are_l2_normalized():
    embedder = HashingEmbedder(dimension=32)
    vector = embedder.embed(["some legal text about statutes"])[0]
    norm = math.sqrt(sum(v * v for v in vector))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_hashing_embedder_empty_text_is_zero_vector():
    embedder = HashingEmbedder(dimension=16)
    vector = embedder.embed([""])[0]
    assert vector == [0.0] * 16


def test_hashing_embedder_rejects_non_positive_dimension():
    with pytest.raises(ValueError):
        HashingEmbedder(dimension=0)


def test_get_embedder_defaults_to_hashing(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    embedder = get_embedder()
    assert isinstance(embedder, HashingEmbedder)


def test_get_embedder_unknown_name_raises():
    with pytest.raises(ValueError):
        get_embedder("not-a-real-provider")


def test_sentence_transformer_embedder_missing_package_raises_clear_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="sentence-transformers"):
        SentenceTransformerEmbedder()
