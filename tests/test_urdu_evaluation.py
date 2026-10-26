"""Urdu / Roman-Urdu retrieval evaluation set (issue #31).

The seed corpus is tiny (3 documents), so hit-rate here is a coarse regression
guard, not a quality claim. Urdu-script queries have no lexical overlap with the
English corpus; they are tracked but not gated until a multilingual embedder lands.
"""
import json
from pathlib import Path

import pytest

from legal_core.evaluation import evaluate_retrieval, load_cases
from legal_core.ingest import run_ingestion
from legal_core.language import detect_language
from legal_core.retrieval import build_retriever

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "evaluation" / "urdu_queries.json"
ROWS = json.loads(CASES.read_text(encoding="utf-8"))


def test_set_shape_and_labels_match_detector():
    assert len(ROWS) >= 10
    assert {r["language"] for r in ROWS} == {"urdu", "roman_urdu"}
    sources = {p.name for p in (ROOT / "data" / "corpus").glob("*.txt")}
    for row in ROWS:
        assert row["expected_source"] in sources
        assert detect_language(row["query"]) == row["language"], row["query"]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    index = tmp_path_factory.mktemp("urdu-index")
    run_ingestion(ROOT / "data" / "corpus", index, embedding_provider="hashing")
    retriever = build_retriever(index, top_k=3, embedding_provider="hashing")
    return evaluate_retrieval(retriever, load_cases(CASES), top_k=3, analyze=True)


def test_roman_urdu_hit_rate_floor(report):
    roman = [c for c, r in zip(report.cases, ROWS) if r["language"] == "roman_urdu"]
    assert sum(c.hit for c in roman) / len(roman) >= 0.7
