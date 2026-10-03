# Improvement Plan

Status legend: [x] done, [ ] open.

## P0 - Fix what is broken
- [x] 1. Evaluation harness compared basenames to absolute paths (reported 0% hit rate) - normalize sources.
- [x] 2. `SearchService` test stub drifted from `CorpusSearchService` (`filters` kwarg).
- [x] 3. "What does section 5 ..." classified as `legal_definition` instead of `statute_lookup`.
- [x] 4. CI: install package, lint with ruff, coverage gate.
- [x] 5. README run command (`legal_core.api:app`).

## P1 - Product quality
- [ ] 6. Real LLM provider (Anthropic) behind `get_llm_provider`, offline default kept.
- [ ] 7. Better embeddings: improved offline tokenizer, optional sentence-transformers extra.
- [ ] 8. Larger evaluation set with out-of-corpus queries, recall@k, MRR, no-source precision.
- [ ] 9. Public-source ingestion adapters with license tracking.
- [ ] 10. Minimal web UI (ask, search, filters, citations, disclaimer).

## P2 - Hardening
- [ ] 11. Security defaults: refuse the dev API key in production, trusted-proxy client IP.
- [ ] 12. Observability: structured logs with request IDs and latency, no raw queries by default.
- [ ] 13. Docs cleanup (README status, API section, architecture module names).
- [ ] 14. Index versioning / manifest and incremental re-ingestion.

## P3 - Roadmap (PRD phases 2-4)
Urdu / Roman Urdu, accounts, saved searches, uploads + OCR, case similarity, citation graph.
