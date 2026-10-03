# Architecture Notes: Pakistani Lawyer AI

## Pipeline

```text
Corpus (public statutes/judgments, .txt/.pdf)
    -> Ingestion (chunk + embed, source citation kept per chunk)
    -> Vector store
User question -> Retrieve top-k chunks -> LLM answer grounded in retrieved text
    -> Answer + cited section/judgment references
```

## Components

- `ingest` / `chunking` / `statute_structure` / `metadata` — section-aware chunking
  (chunks start at `Section N` / `Article N` headings), per-chunk citation and legal
  metadata, idempotent re-ingestion, pruning of stale chunks, `manifest.json` index record
- `sources` — licensed public-source acquisition (license allowlist, provenance)
- `embeddings` / `embedding_pipeline` / `vector_store` — pluggable embedders
  (offline hashing default, optional sentence-transformers) and a local cosine store
- `retrieval` / `hybrid_retrieval` / `filters` / `authority` — vector, BM25 and fused
  retrieval with metadata filters and authority-aware reranking; fused scores are on an
  absolute [0, 1] scale so the relevance floor is meaningful
- `query_analysis` / `service` — intent + jurisdiction analysis, relevance gate,
  generation with one retry, citation verification, no-source fallback
- `llm` / `generation` / `qa` / `citation_verification` — provider interface (local,
  Anthropic), grounded context, citation objects, inline-marker verification
- `api` / `security` / `static/index.html` — FastAPI app, auth, rate limits, trusted-proxy
  client identity, JSON access logs, web UI
- `evaluation` — precision@k, hit rate, MRR, section recall, no-source precision

## Design Notes

- Re-ingestion is idempotent by deterministic chunk id; chunks whose source changed or
  disappeared are pruned so superseded law never lingers in the index.
- Keep the LLM provider swappable behind an interface (see `multi-llm-router`
  in this portfolio for the general provider-swap pattern); default to a fake
  backend so the project runs offline with no API key.
- Never present an answer without a citation or an explicit "no relevant
  source found" signal — a grounded answer with no retrieved support is a
  silent hallucination risk in a legal-information context.
- Keep the corpus small and clearly public-domain; do not scrape or ingest
  copyrighted case-law text.

## Consolidation note

This project is standalone and does not consolidate any other repo.
