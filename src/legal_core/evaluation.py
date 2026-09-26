"""Reproducible retrieval and citation evaluation harness (issue #14)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .generation import AnswerGenerator
from .ingest import run_ingestion
from .qa import answer_from_chunks
from .retrieval import Retriever, build_retriever

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "data" / "evaluation" / "retrieval_queries.json"
DEFAULT_DOC = ROOT / "docs" / "evaluation.md"


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected_source: str


@dataclass(frozen=True)
class RetrievalCaseResult:
    query: str
    expected_source: str
    retrieved_sources: tuple[str | None, ...]
    precision_at_k: float
    hit: bool


@dataclass(frozen=True)
class CitationSample:
    query: str
    answer: str
    citations: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    top_k: int
    mean_precision_at_k: float
    hit_rate_at_k: float
    cases: tuple[RetrievalCaseResult, ...]
    citation_samples: tuple[CitationSample, ...]


def load_cases(path: str | Path = DEFAULT_CASES) -> list[EvaluationCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation query set must be a non-empty list")
    return [
        EvaluationCase(
            query=str(row["query"]).strip(),
            expected_source=str(row["expected_source"]).strip(),
        )
        for row in rows
    ]


def evaluate_retrieval(
    retriever: Retriever,
    cases: list[EvaluationCase],
    *,
    top_k: int = 3,
    citation_sample_size: int = 3,
) -> EvaluationReport:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    case_results = []
    samples = []
    generator = AnswerGenerator()

    for case in cases:
        chunks = retriever.retrieve(case.query, top_k=top_k)
        sources = tuple(chunk.source for chunk in chunks)
        relevant = sum(source == case.expected_source for source in sources)
        precision = relevant / top_k
        hit = relevant > 0

        case_results.append(
            RetrievalCaseResult(
                query=case.query,
                expected_source=case.expected_source,
                retrieved_sources=sources,
                precision_at_k=precision,
                hit=hit,
            )
        )

        if chunks and len(samples) < citation_sample_size:
            answer = answer_from_chunks(case.query, chunks, generator=generator)
            samples.append(
                CitationSample(
                    query=case.query,
                    answer=answer.answer,
                    citations=tuple(c.citation for c in answer.citations),
                )
            )

    count = len(case_results)
    mean_precision = sum(row.precision_at_k for row in case_results) / count
    hit_rate = sum(row.hit for row in case_results) / count

    return EvaluationReport(
        top_k=top_k,
        mean_precision_at_k=mean_precision,
        hit_rate_at_k=hit_rate,
        cases=tuple(case_results),
        citation_samples=tuple(samples),
    )


def result_log_rows(report: EvaluationReport, *, run_date: str | None = None) -> str:
    run_date = run_date or date.today().isoformat()
    return "\n".join(
        [
            f"| {run_date} | Phase 5 | Retrieval precision@{report.top_k} | "
            f"{report.mean_precision_at_k:.3f} | {len(report.cases)} labeled queries |",
            f"| {run_date} | Phase 5 | Retrieval hit rate@{report.top_k} | "
            f"{report.hit_rate_at_k:.3f} | expected source present in top-k |",
        ]
    )


def append_result_log(
    report: EvaluationReport,
    docs_path: str | Path = DEFAULT_DOC,
    *,
    run_date: str | None = None,
) -> None:
    path = Path(docs_path)
    content = path.read_text(encoding="utf-8")
    marker = "| _TBD_ | | | | |"
    rows = result_log_rows(report, run_date=run_date)
    if marker in content:
        content = content.replace(marker, rows, 1)
    else:
        content = content.rstrip() + "\n" + rows + "\n"
    path.write_text(content, encoding="utf-8")


def render_report(report: EvaluationReport) -> str:
    lines = [
        f"Retrieval precision@{report.top_k}: {report.mean_precision_at_k:.3f}",
        f"Retrieval hit rate@{report.top_k}: {report.hit_rate_at_k:.3f}",
        "",
        "Citation spot-check samples:",
    ]
    for sample in report.citation_samples:
        lines.append(f"- Query: {sample.query}")
        lines.append(f"  Citations: {', '.join(sample.citations) or 'none'}")
        lines.append(f"  Answer: {' '.join(sample.answer.split())[:240]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate legal retrieval and citation grounding.")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--index-dir", default="data/evaluation-index")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--citation-samples", type=int, default=3)
    parser.add_argument("--write-docs", action="store_true")
    args = parser.parse_args(argv)

    run_ingestion(args.corpus, args.index_dir, embedding_provider="hashing")
    retriever = build_retriever(
        args.index_dir,
        top_k=args.top_k,
        embedding_provider="hashing",
    )
    report = evaluate_retrieval(
        retriever,
        load_cases(args.cases),
        top_k=args.top_k,
        citation_sample_size=max(0, args.citation_samples),
    )
    print(render_report(report))
    if args.write_docs:
        append_result_log(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
