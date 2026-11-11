# Evaluation Notes: Pakistani Lawyer AI

## Metrics

- **Retrieval precision@k** — for the hand-labeled query set in
  `data/evaluation/retrieval_queries.json`, the fraction of the first k
  retrieved chunks that come from the expected source.
- **Retrieval hit rate@k** — the fraction of labeled queries where the expected
  source appears at least once in the first k results.
- **Citation accuracy (manual spot check)** — inspect the citation samples
  emitted by the evaluation harness and verify that each generated/extractive
  answer is actually supported by the displayed source chunks.
- **"No relevant source" precision** — for out-of-corpus questions, the
  fraction of responses that correctly say no relevant source was found
  instead of fabricating an answer.

## Running the evaluation

The evaluation uses the checked-in seed corpus and the offline hashing
embedder, so it requires no network access or paid model.

```bash
python -m legal_core.evaluation --top-k 3
```

To append the measured retrieval metrics to the Result Log:

```bash
python -m legal_core.evaluation --top-k 3 --write-docs
```

The command also prints a small citation spot-check sample for manual review.

## Corpus licensing note

The seed corpus under `data/corpus/` is original sample statute-style text
authored for this project, not a reproduction of any real statute.
`data/corpus/sources.json` records the source/license basis per document.

## Result Log

| Date | Phase | Metric | Value | Notes |
|------|-------|--------|-------|-------|
| _TBD_ | | | | |

## Release and QA safety notes

The sample corpus is fictional. Its passing tests **do not** establish accuracy
against Pakistani law. Do not deploy it as a legal advice service. The production
`APP_ENV` gate blocks startup with the sample corpus or documents without
`source_url` metadata; provenance URLs alone do not prove current legal validity.

Citation integrity verifies that answer text references retrieved source labels
rather than relying on citations attached automatically by the application.
**It does not verify factual entailment, whether a judgment is binding, or
whether a statute is currently in force.** Advocate review and source/version
verification remain necessary before real-user deployment.

The Docker image rebuilds the search index at startup after persistent volumes
have mounted. `GET /health` confirms process health; `GET /ready` returns
HTTP 503 when the indexed corpus is missing, empty, or corrupt.

For a local smoke test (after installing development requirements):

```bash
pytest tests/test_integration_smoke.py
pytest tests/
ruff check src tests
```

Remaining work is tracked under GitHub issues #28-#34. In particular, real
legislation, a lawyer-reviewed evaluation set, account-level authorization,
privacy controls, and security testing are release blockers for public use.
