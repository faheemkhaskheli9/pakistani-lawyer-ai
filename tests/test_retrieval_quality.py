"""Offline retrieval-quality smoke tests for the checked-in seed corpus.

The labeled query set is intentionally small and deterministic.  Its purpose
is to catch obvious retrieval regressions before answer generation is tested.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_core.ingest import run_ingestion
from legal_core.retrieval import build_retriever

ROOT = Path(__file__).resolve().parents[1]
SEED_CORPUS = ROOT / "data" / "corpus"
LABELED_QUERIES = ROOT / "data" / "evaluation" / "retrieval_queries.json"


def load_labeled_queries() -> list[dict[str, str]]:
    rows = json.loads(LABELED_QUERIES.read_text(encoding="utf-8"))
    assert len(rows) >= 5
    for row in rows:
        assert set(row) == {"query", "expected_source"}
        assert row["query"].strip()
        assert row["expected_source"].strip()
    return rows


@pytest.fixture(scope="module")
def labeled_queries() -> list[dict[str, str]]:
    return load_labeled_queries()


def test_labeled_query_set_covers_each_seed_document(labeled_queries):
    expected_sources = {row["expected_source"] for row in labeled_queries}
    seed_sources = {path.name for path in SEED_CORPUS.glob("*.txt")}
    assert expected_sources == seed_sources


def test_expected_source_is_returned_in_top_k(tmp_path, labeled_queries):
    index_dir = tmp_path / "index"
    ingested, _ = run_ingestion(
        SEED_CORPUS,
        index_dir,
        embedding_provider="hashing",
    )
    assert ingested > 0

    retriever = build_retriever(
        index_dir,
        top_k=5,
        embedding_provider="hashing",
    )

    failures = []
    for row in labeled_queries:
        results = retriever.retrieve(row["query"])
        sources = [result.source for result in results]
        if row["expected_source"] not in sources:
            failures.append(
                {
                    "query": row["query"],
                    "expected": row["expected_source"],
                    "retrieved": sources,
                }
            )

    assert failures == []
