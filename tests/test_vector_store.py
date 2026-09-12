"""Tests for issue #2's local vector store: persistence, reload without
re-embedding, upsert-is-idempotent, and cosine-similarity query."""

from __future__ import annotations

import pytest

from legal_core.vector_store import DimensionMismatch, VectorStore


def _unit(*values: float) -> list[float]:
    import math

    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def test_upsert_and_query_finds_the_closest_match(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.upsert("a", _unit(1, 0, 0), "doc about contracts")
    store.upsert("b", _unit(0, 1, 0), "doc about criminal law")

    results = store.query(_unit(0.9, 0.1, 0), top_k=1)

    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].document == "doc about contracts"


def test_query_returns_up_to_top_k_ordered_by_score_desc(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.upsert("a", _unit(1, 0), "a")
    store.upsert("b", _unit(0.9, 0.1), "b")
    store.upsert("c", _unit(0, 1), "c")

    results = store.query(_unit(1, 0), top_k=2)

    assert [r.id for r in results] == ["a", "b"]
    assert results[0].score >= results[1].score


def test_upsert_same_id_again_overwrites_not_duplicates(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.upsert("a", _unit(1, 0), "first version")
    store.upsert("a", _unit(0, 1), "second version")

    assert len(store) == 1
    assert "a" in store
    results = store.query(_unit(0, 1), top_k=1)
    assert results[0].document == "second version"


def test_store_persists_and_reloads_without_re_embedding(tmp_path):
    path = tmp_path / "store"
    store = VectorStore(path)
    store.upsert("a", _unit(1, 0, 0), "doc a", metadata={"source": "s1"})

    reloaded = VectorStore(path)

    assert len(reloaded) == 1
    assert "a" in reloaded
    results = reloaded.query(_unit(1, 0, 0), top_k=1)
    assert results[0].document == "doc a"
    assert results[0].metadata == {"source": "s1"}


def test_dimension_mismatch_raises_instead_of_corrupting_the_store(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.upsert("a", _unit(1, 0, 0), "doc a")

    with pytest.raises(DimensionMismatch):
        store.upsert("b", [1.0, 0.0], "doc b")


def test_query_on_empty_store_returns_no_results(tmp_path):
    store = VectorStore(tmp_path / "store")
    assert store.query(_unit(1, 0), top_k=5) == []


def test_interrupted_save_never_corrupts_a_previously_persisted_store(tmp_path, monkeypatch):
    path = tmp_path / "store"
    store = VectorStore(path)
    store.upsert("a", _unit(1, 0), "doc a")

    reopened = VectorStore(path)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    import json as json_module

    monkeypatch.setattr(json_module, "dump", boom)
    with pytest.raises(OSError):
        reopened.upsert("b", _unit(0, 1), "doc b")

    # The on-disk store must still be the last good version, not a
    # truncated/partial one.
    recovered = VectorStore(path)
    assert len(recovered) == 1
    assert "a" in recovered
    assert "b" not in recovered
