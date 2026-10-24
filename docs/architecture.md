# Architecture Notes: Pakistani Lawyer AI

> Product is a web application. Full designs: [PRD](PRD.md), [TRD](TRD.md), [System Design](system-design.md). This file keeps the short pipeline summary.

## Pipeline

```text
Corpus (public statutes/judgments, .txt/.pdf)
    -> Ingestion (chunk + embed, source citation kept per chunk)
    -> Vector store
User question -> Retrieve top-k chunks -> LLM answer grounded in retrieved text
    -> Answer + cited section/judgment references
```

## Components

- `legal_core.ingest` — chunking + embedding, idempotent re-ingestion
  (re-running on an unchanged corpus does not duplicate vectors), per-chunk
  source citation (statute name + section, or judgment + paragraph)
- `legal_core.retrieval` / `hybrid_retrieval` — top-k vector and BM25 hybrid retrieval with filters
- `legal_core.service` / `generation` / `citation_verification` — grounded LLM answer,
  citation verification, "no relevant source found" fallback
- API layer (FastAPI) — `/api/v1/ask`, `/api/v1/ask/stream` (SSE), `/api/v1/search`
- Web frontend (Next.js) — planned; consumes the API (see TRD)

## Design Notes

- Apply this portfolio's `rag/chunking-and-embedding-ingestion` knowledge-base
  pattern for idempotent re-ingestion and source citation — do not
  re-implement the ingestion contract from scratch.
- Keep the LLM provider swappable behind an interface (see `multi-llm-router`
  in this portfolio for the general provider-swap pattern); default to a fake
  backend so the project runs offline with no API key.
- Never present an answer without a citation or an explicit "no relevant
  source found" signal — a grounded answer with no retrieved support is a
  silent hallucination risk in a legal-information context.
- Keep the corpus small and clearly public-domain; do not scrape or ingest
  copyrighted case-law text.

## Consolidation note

This project is standalone — it does not consolidate any other portfolio
repo. `docs/architecture.md` exists here purely to document the pipeline and
knowledge-base pattern references for whoever implements Phase 1.
