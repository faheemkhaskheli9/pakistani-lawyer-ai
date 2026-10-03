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
    assert len(rows) >= 20
    for row in rows:
        assert {"query", "expected_source"} <= set(row) <= {"query", "expected_source", "expected_section"}
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
        sources = [Path(result.source).name for result in results]
        if row["expected_source"] not in sources:
            failures.append(
                {
                    "query": row["query"],
                    "expected": row["expected_source"],
                    "retrieved": sources,
                }
            )

    assert failures == []


def test_hybrid_quality_floor_and_no_source_refusals(tmp_path):
    from legal_core.evaluation import (
        evaluate_no_source,
        evaluate_retrieval,
        load_cases,
        load_out_of_corpus,
    )
    from legal_core.hybrid_retrieval import build_hybrid_retriever
    from legal_core.service import QuestionAnsweringService

    run_ingestion(SEED_CORPUS, tmp_path / "idx", embedding_provider="hashing")
    retriever = build_hybrid_retriever(tmp_path / "idx", top_k=3, embedding_provider="hashing")
    report = evaluate_retrieval(retriever, load_cases(LABELED_QUERIES), top_k=3)
    assert report.hit_rate_at_k == 1.0
    assert report.mrr >= 0.95
    assert report.section_recall_at_k >= 0.85

    service = QuestionAnsweringService(retriever)
    assert evaluate_no_source(service, load_out_of_corpus()) >= 0.9
