"""Offline benchmark tests; none certify the actual content of Pakistani law."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from legal_core.chunk_store import ChunkRecord, ChunkStore
from legal_core.qa import AnswerCitation, LegalAnswer
from legal_core.safety_benchmark import (
    DEFAULT_CASES,
    BenchmarkCase,
    BenchmarkDataset,
    check_source_snapshots,
    evaluate_safety,
    load_dataset,
    main,
)
from legal_core.service import QAResult

ROOT = Path(__file__).resolve().parents[1]


def case(
    case_id: str, question: str, status: str,
    *, language: str = "english", sources: tuple[str, ...] = (),
) -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        question=question,
        language=language,
        category="grounded" if status == "answered" else "out_of_corpus",
        expected_status=status,
        expected_sources=sources,
        review_status="synthetic",
    )


def make_dataset(*cases: BenchmarkCase) -> BenchmarkDataset:
    return BenchmarkDataset("test-fixture", "synthetic", "a" * 64, tuple(cases))


def write_dataset(tmp_path, row, *, kind="synthetic"):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({
        "schema_version": 1, "dataset_id": "fixture", "kind": kind, "cases": [row],
    }), encoding="utf-8")
    return path


def row(*, status="synthetic"):
    return {
        "id": "case-one", "question": "What is in the fictional document?",
        "language": "english", "category": "grounded",
        "expected_status": "answered", "expected_sources": ["law.txt"],
        "review": {"status": status},
    }


def test_checked_in_120_cases_are_only_synthetic():
    dataset = load_dataset(DEFAULT_CASES)
    assert len(dataset.cases) == 120
    assert dataset.kind == "synthetic"
    assert all(c.review_status == "synthetic" for c in dataset.cases)
    assert len({c.id for c in dataset.cases}) == 120
    assert {c.language for c in dataset.cases} == {"english", "roman_urdu", "urdu"}
    assert {c.category for c in dataset.cases} == {
        "grounded", "out_of_corpus", "wrong_jurisdiction", "prompt_injection",
    }
    assert sum(c.expected_status == "answered" for c in dataset.cases) == 72


def test_synthetic_cases_cannot_be_misrepresented_as_lawyer_approved():
    with pytest.raises(ValueError, match="synthetic cases cannot"):
        load_dataset(DEFAULT_CASES, require_approved=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"id": "../../unsafe"}, "invalid case id"),
        ({"language": "french"}, "unsupported language"),
        ({"category": "legal_advice"}, "unsupported language or category"),
        ({"expected_status": "arbitrary"}, "unsupported expected_status"),
        ({"expected_sources": ["../law.txt"]}, "expected_sources"),
        ({"expected_sources": []}, "answerable cases require sources"),
        ({"review": {"status": "approved"}}, "invalid review status"),
    ],
)
def test_invalid_label_data_is_rejected(tmp_path, mutation, message):
    data = row()
    data.update(mutation)
    with pytest.raises(ValueError, match=message):
        load_dataset(write_dataset(tmp_path, data))


def test_duplicate_identifiers_are_rejected(tmp_path):
    p = write_dataset(tmp_path, row())
    obj = json.loads(p.read_text())
    obj["cases"].append(row())
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="duplicate or invalid"):
        load_dataset(p)


def test_approved_legal_review_requires_reviewer_evidence_and_content_hash(tmp_path):
    data = row(status="approved")
    with pytest.raises(ValueError, match="reviewer"):
        load_dataset(write_dataset(tmp_path, data, kind="legal"), require_approved=True)

    data["review"] = {
        "status": "approved", "reviewer": "lawyer reviewer",
        "reviewed_on": date.today().isoformat(), "evidence_ref": "case-law review record",
    }
    with pytest.raises(ValueError, match="corpus_sha256"):
        load_dataset(write_dataset(tmp_path, data, kind="legal"), require_approved=True)

    data["source_hashes"] = {"law.txt": "f" * 64}
    fixture = json.loads((tmp_path / "cases.json").read_text())
    fixture["corpus_sha256"] = "e" * 64
    fixture["cases"] = [data]
    (tmp_path / "cases.json").write_text(json.dumps(fixture), encoding="utf-8")
    ds = load_dataset(tmp_path / "cases.json", require_approved=True)
    assert ds.kind == "legal" and ds.cases[0].review_status == "approved"


def test_unapproved_legal_cases_do_not_pass_approved_gate(tmp_path):
    with pytest.raises(ValueError, match="not approved"):
        load_dataset(write_dataset(tmp_path, row(status="draft"), kind="legal"), require_approved=True)


def test_snapshot_hash_prevents_reusing_approval_after_amendment(tmp_path):
    record = ChunkRecord(
        id="a", source="/private/law.txt", section="Section 1", chunk_index=0,
        text="fixture", metadata={"source_sha256": "1" * 64},
    )
    index = tmp_path / "index"
    ChunkStore(index / "chunks.json").upsert([record])
    approved = replace(
        case("approved-case", "Question", "answered", sources=("law.txt",)),
        review_status="approved", source_hashes={"law.txt": "2" * 64},
    )
    with pytest.raises(ValueError, match="differs from reviewed snapshot"):
        check_source_snapshots(make_dataset(approved), index)

    check_source_snapshots(
        make_dataset(replace(approved, source_hashes={"law.txt": "1" * 64})),
        index,
    )


class StubService:
    def ask(self, question):
        if question == "backend-crash":
            raise RuntimeError("backend unavailable")
        if question == "negative-ok":
            return QAResult(status="no_relevant_source", message="No source.", answer=None)
        src = "wrong.txt" if question == "source-wrong" else "law.txt"
        citation = AnswerCitation(
            chunk_id="a", source=src, section="Section 1",
            citation=f"{src}, Section 1", score=0.9, excerpt="Fixture.",
        )
        answer = LegalAnswer(question=question, answer="Sample [SOURCE 1: law.txt]", citations=(citation,))
        return QAResult(
            status="answered", message="Fixture.", answer=answer,
            citation_verification=SimpleNamespace(verified=True),
        )


def test_metrics_separate_unsafe_answers_from_withheld_answers_and_errors():
    dataset = make_dataset(
        case("positive-one", "positive", "answered", sources=("law.txt",)),
        case("positive-two", "source-wrong", "answered", sources=("law.txt",)),
        case("negative-one", "negative-ok", "no_relevant_source"),
        case("negative-two", "answer-bad", "no_relevant_source"),
        case("negative-three", "backend-crash", "no_relevant_source"),
    )
    report = evaluate_safety(StubService(), dataset)
    metrics = report["metrics"]
    assert metrics["count"] == 5
    assert metrics["passed"] == 2
    assert metrics["pass_rate"] == 0.4
    assert metrics["source_hit_rate"] == 0.5
    assert metrics["false_answer_rate"] == pytest.approx(1 / 3)
    assert metrics["correct_refusal_rate"] == pytest.approx(1 / 3)
    assert metrics["error_count"] == 1
    assert report["legal_accuracy_established"] is False
    assert report["cases"][-1]["error"].startswith("RuntimeError:")


def test_empty_denominator_is_reported_as_null_not_as_zero():
    result = evaluate_safety(
        StubService(),
        make_dataset(case("positive-one", "positive", "answered", sources=("law.txt",))),
    )
    assert result["metrics"]["false_answer_rate"] is None


def test_cli_runs_real_local_service_and_exports_machine_readable_report(tmp_path, capsys):
    output = tmp_path / "report.json"
    code = main([
        "--corpus", str(ROOT / "data" / "corpus"),
        "--index-dir", str(tmp_path / "index"),
        "--json-out", str(output),
        "--top-k", "3",
    ])
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["case_count"] == 120
    assert report["dataset_kind"] == "synthetic"
    assert report["approved_cases"] == 0
    assert report["legal_accuracy_established"] is False
    assert len(report["cases"]) == 120
    assert all(0 <= row["metrics"]["pass_rate"] <= 1 for row in [
        {"metrics": report["metrics"]}, *[
            {"metrics": summary} for summary in report["by_language"].values()
        ],
    ])
    assert "dataset_sha256" in capsys.readouterr().out


def test_cli_rejects_approval_gate_without_mutating_index(tmp_path):
    index = tmp_path / "index"
    rc = main([
        "--cases", str(DEFAULT_CASES),
        "--corpus", str(ROOT / "data" / "corpus"),
        "--index-dir", str(index),
        "--require-approved",
    ])
    assert rc == 1
    assert not index.exists()

def test_approved_abstention_label_requires_full_corpus_snapshot(tmp_path):
    negative = row(status="approved")
    negative["expected_status"] = "no_relevant_source"
    negative["expected_sources"] = []
    negative["review"] = {
        "status": "approved", "reviewer": "expert",
        "reviewed_on": date.today().isoformat(), "evidence_ref": "review fixture",
    }
    path = write_dataset(tmp_path, negative, kind="legal")
    with pytest.raises(ValueError, match="corpus_sha256"):
        load_dataset(path, require_approved=True)


def test_complete_corpus_fingerprint_changes_when_an_unrelated_source_changes(tmp_path):
    from legal_core.safety_benchmark import corpus_snapshot_digest

    index = tmp_path / "index"
    store = ChunkStore(index / "chunks.json")
    one = ChunkRecord(
        id="a", source="law.txt", section="Section 1", chunk_index=0,
        text="source a", metadata={"source_sha256": "a" * 64},
    )
    other = ChunkRecord(
        id="b", source="other.txt", section="Section 1", chunk_index=0,
        text="source b", metadata={"source_sha256": "b" * 64},
    )
    store.upsert([one, other])
    old = corpus_snapshot_digest(index)
    store.upsert([replace(other, metadata={"source_sha256": "c" * 64})])
    new = corpus_snapshot_digest(index)
    assert old != new
    assert len(old) == len(new) == 64
