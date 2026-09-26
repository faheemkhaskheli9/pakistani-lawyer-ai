"""Tests for the evaluation harness (issue #14)."""
from __future__ import annotations

from legal_core.evaluation import (
    EvaluationCase,
    append_result_log,
    evaluate_retrieval,
    result_log_rows,
)
from legal_core.retrieval import RetrievedChunk


class StubRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def retrieve(self, query, top_k=None):
        return self.mapping[query][:top_k]


def chunk(rank, source, chunk_id):
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        score=1.0 - rank * 0.1,
        text=f"Source text from {source}.",
        source=source,
        section=f"Section {rank}",
        chunk_index=rank - 1,
    )


def test_precision_and_hit_rate_are_computed_from_labeled_sources():
    cases = [
        EvaluationCase("q1", "expected-a.txt"),
        EvaluationCase("q2", "expected-b.txt"),
    ]
    retriever = StubRetriever(
        {
            "q1": [
                chunk(1, "expected-a.txt", "a1"),
                chunk(2, "other.txt", "a2"),
            ],
            "q2": [
                chunk(1, "other.txt", "b1"),
                chunk(2, "expected-b.txt", "b2"),
            ],
        }
    )

    report = evaluate_retrieval(retriever, cases, top_k=2)

    assert report.mean_precision_at_k == 0.5
    assert report.hit_rate_at_k == 1.0
    assert len(report.citation_samples) == 2
    assert report.citation_samples[0].citations


def test_missed_source_reduces_hit_rate():
    cases = [EvaluationCase("q", "expected.txt")]
    retriever = StubRetriever({"q": [chunk(1, "other.txt", "x")]})
    report = evaluate_retrieval(retriever, cases, top_k=1)
    assert report.mean_precision_at_k == 0.0
    assert report.hit_rate_at_k == 0.0


def test_result_log_format_matches_evaluation_document_table():
    report = evaluate_retrieval(
        StubRetriever({"q": [chunk(1, "expected.txt", "x")]}),
        [EvaluationCase("q", "expected.txt")],
        top_k=1,
        citation_sample_size=0,
    )
    rows = result_log_rows(report, run_date="2026-09-26")
    assert "| 2026-09-26 | Phase 5 | Retrieval precision@1 | 1.000 |" in rows
    assert "Retrieval hit rate@1" in rows


def test_append_result_log_replaces_placeholder(tmp_path):
    path = tmp_path / "evaluation.md"
    path.write_text(
        "# Evaluation\n\n| Date | Phase | Metric | Value | Notes |\n"
        "|---|---|---|---|---|\n| _TBD_ | | | | |\n",
        encoding="utf-8",
    )
    report = evaluate_retrieval(
        StubRetriever({"q": [chunk(1, "expected.txt", "x")]}),
        [EvaluationCase("q", "expected.txt")],
        top_k=1,
        citation_sample_size=0,
    )

    append_result_log(report, path, run_date="2026-09-26")

    text = path.read_text(encoding="utf-8")
    assert "_TBD_" not in text
    assert "Retrieval precision@1" in text
