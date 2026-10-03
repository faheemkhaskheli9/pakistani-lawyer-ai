"""Reproducible retrieval and citation evaluation harness (issue #14)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from .generation import AnswerGenerator
from .hybrid_retrieval import build_hybrid_retriever
from .ingest import run_ingestion
from .qa import answer_from_chunks
from .retrieval import Retriever, build_retriever
from .service import QuestionAnsweringService

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "data" / "evaluation" / "retrieval_queries.json"
DEFAULT_OOC = ROOT / "data" / "evaluation" / "out_of_corpus_queries.json"
DEFAULT_DOC = ROOT / "docs" / "evaluation.md"


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected_source: str
    expected_section: str | None = None


@dataclass(frozen=True)
class RetrievalCaseResult:
    query: str
    expected_source: str
    retrieved_sources: tuple[str | None, ...]
    precision_at_k: float
    hit: bool
    reciprocal_rank: float = 0.0
    section_hit: bool | None = None


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
    mrr: float = 0.0
    section_recall_at_k: float | None = None
    no_source_precision: float | None = None
    no_source_total: int = 0


def source_name(source: str | None) -> str | None:
    """Normalize a retrieved source (usually a file path) to its file name."""
    if not source:
        return None
    return Path(source).name


def load_cases(path: str | Path = DEFAULT_CASES) -> list[EvaluationCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation query set must be a non-empty list")
    return [
        EvaluationCase(
            query=str(row["query"]).strip(),
            expected_source=str(row["expected_source"]).strip(),
            expected_section=(str(row["expected_section"]).strip() if row.get("expected_section") else None),
        )
        for row in rows
    ]


def load_out_of_corpus(path: str | Path = DEFAULT_OOC) -> list[str]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(row).strip() for row in rows if str(row).strip()]


def evaluate_no_source(service, queries: list[str]) -> float:
    """Fraction of out-of-corpus queries correctly refused with no relevant source."""
    if not queries:
        raise ValueError("at least one out-of-corpus query is required")
    refused = sum(service.ask(q).status == "no_relevant_source" for q in queries)
    return refused / len(queries)


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
        sources = tuple(source_name(chunk.source) for chunk in chunks)
        relevant = sum(source == case.expected_source for source in sources)
        precision = relevant / top_k
        hit = relevant > 0
        rr = next((1.0 / i for i, src in enumerate(sources, 1) if src == case.expected_source), 0.0)
        section_hit = None
        if case.expected_section:
            section_hit = any(
                source_name(c.source) == case.expected_source and c.section == case.expected_section
                for c in chunks
            )

        case_results.append(
            RetrievalCaseResult(
                query=case.query,
                expected_source=case.expected_source,
                retrieved_sources=sources,
                precision_at_k=precision,
                hit=hit,
                reciprocal_rank=rr,
                section_hit=section_hit,
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
    mrr = sum(row.reciprocal_rank for row in case_results) / count
    sectioned = [row for row in case_results if row.section_hit is not None]
    section_recall = sum(row.section_hit for row in sectioned) / len(sectioned) if sectioned else None

    return EvaluationReport(
        top_k=top_k,
        mean_precision_at_k=mean_precision,
        hit_rate_at_k=hit_rate,
        cases=tuple(case_results),
        citation_samples=tuple(samples),
        mrr=mrr,
        section_recall_at_k=section_recall,
    )


def result_log_rows(report: EvaluationReport, *, run_date: str | None = None) -> str:
    run_date = run_date or date.today().isoformat()
    return "\n".join(
        [
            f"| {run_date} | Phase 5 | Retrieval precision@{report.top_k} | "
            f"{report.mean_precision_at_k:.3f} | {len(report.cases)} labeled queries |",
            f"| {run_date} | Phase 5 | Retrieval hit rate@{report.top_k} | "
            f"{report.hit_rate_at_k:.3f} | expected source present in top-k |",
            f"| {run_date} | Phase 5 | MRR | {report.mrr:.3f} | source-level reciprocal rank |",
        ]
        + (
            [
                f"| {run_date} | Phase 5 | Section recall@{report.top_k} | "
                f"{report.section_recall_at_k:.3f} | expected section present in top-k |"
            ]
            if report.section_recall_at_k is not None
            else []
        )
        + (
            [
                f"| {run_date} | Phase 5 | No-source precision | {report.no_source_precision:.3f} | "
                f"{report.no_source_total} out-of-corpus queries refused |"
            ]
            if report.no_source_precision is not None
            else []
        )
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
        f"MRR: {report.mrr:.3f}",
        *(
            [f"Section recall@{report.top_k}: {report.section_recall_at_k:.3f}"]
            if report.section_recall_at_k is not None
            else []
        ),
        *(
            [f"No-source precision: {report.no_source_precision:.3f} ({report.no_source_total} queries)"]
            if report.no_source_precision is not None
            else []
        ),
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
    parser.add_argument("--out-of-corpus", default=str(DEFAULT_OOC))
    parser.add_argument("--retriever", choices=("vector", "hybrid"), default="hybrid")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--embedding-provider",
        default="hashing",
        help="hashing (offline default) or sentence-transformers (needs the 'embeddings' extra). "
        "Real embeddings use a different score scale, so tune --min-score for them.",
    )
    parser.add_argument("--write-docs", action="store_true")
    args = parser.parse_args(argv)

    run_ingestion(args.corpus, args.index_dir, embedding_provider=args.embedding_provider)
    builder = build_hybrid_retriever if args.retriever == "hybrid" else build_retriever
    retriever = builder(args.index_dir, top_k=args.top_k, embedding_provider=args.embedding_provider)
    report = evaluate_retrieval(
        retriever,
        load_cases(args.cases),
        top_k=args.top_k,
        citation_sample_size=max(0, args.citation_samples),
    )
    ooc = load_out_of_corpus(args.out_of_corpus)
    service = QuestionAnsweringService(retriever, min_score=args.min_score)
    report = replace(
        report,
        no_source_precision=evaluate_no_source(service, ooc),
        no_source_total=len(ooc),
    )
    print(render_report(report))
    if args.write_docs:
        append_result_log(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
