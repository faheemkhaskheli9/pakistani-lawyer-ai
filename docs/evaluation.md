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

## Safety benchmark: 120 synthetic multilingual cases (Phase 2)

A second, **separate** evaluator exercises the real Q&A service rather than
retrieval alone. Its checked-in dataset is
`data/evaluation/synthetic_safety_cases.json` (72 answerable and 48
negative cases across English, Urdu, Roman Urdu, jurisdiction filtering and
prompt injection). Every case is marked `review.status: synthetic`. This
dataset tests software regressions only; it contains **no legally verified
Pakistani advice**.

```bash
python -m legal_core.safety_benchmark \
  --cases data/evaluation/synthetic_safety_cases.json \
  --corpus data/corpus --index-dir data/evaluation-index \
  --json-out data/evaluation-index/synthetic-safety-report.json
```

Output records the dataset SHA-256 for reproducibility, counts, individual
case outcomes (case IDs rather than possibly sensitive prompts), and
breakdowns by language/category.

Metrics:
- `pass_rate`: exact expected status; for answerable cases also requires a
  cited expected source.
- `source_hit_rate`: share of answerable cases where an expected document is
  cited in an `answered` response.
- `false_answer_rate`: share of cases labeled `no_relevant_source` that
  nevertheless produced an `answered` result; lower is safer.
- `correct_refusal_rate`: share of unanswerable cases that explicitly returned
  `no_relevant_source`. A citation-verification failure is measured separately
  as `withheld_answer_rate`, not treated as a correct refusal.
- `citation_integrity_rate`: metadata/inline-label verification among
  `answered` responses; this is **not** semantic factual entailment.
- `error_count`: provider, parsing or other service errors.

A missing denominator is JSON `null`, not zero. Automatic language metrics
include only supplied labels, which may themselves need human verification.

### Lawyer-reviewed evaluation workflow (not yet populated)

1. Build a real corpus using the versioned,
   [rights-cleared source registry](verified-corpus-plan.md), and evaluate it
   in a private, access-controlled environment.
2. Copy `docs/legal_benchmark.example.json` to a private location. Write
   original questions (including no-answer, outdated-statute, cross-province
   and contradictory-precedent cases) with exact source references.
3. Have qualified advocates review each question, its expected status, source
   and legal reasoning independently. Set `review.status` to `approved`
   **only** after review and add reviewer, review date and evidence reference.
4. For answerable cases add `source_hashes` keyed by document basename using
   the SHA-256 of the exact source document. For **all** cases pin the full
   indexed corpus using its digest:

   ```bash
   python -c 'from legal_core.safety_benchmark import corpus_snapshot_digest; print(corpus_snapshot_digest("/srv/legal-index"))'
   ```

   Put that value in the top-level `corpus_sha256`. Any changed, added or
   deleted legal source invalidates the approval snapshot, **including for
   negative (no-answer) labels**.
5. Run the strict benchmark, which refuses synthetic/unapproved labels and
   performs admitted-only corpus ingestion before evaluating:

   ```bash
   python -m legal_core.safety_benchmark \
     --cases /secure/lawyer-reviewed-cases.json \
     --corpus /srv/legal-corpus --index-dir /srv/legal-index \
     --require-approved --max-false-answer-rate 0.0 \
     --json-out /secure/evaluation-report.json
   ```

Optional `--min-pass-rate` and `--max-false-answer-rate` make a run exit
with code 2 when the configured thresholds fail. A schema/index-validation
error exits with code 1. **A green result is not legal certification**:
benchmark labels are human attestations, citation checks validate markers
rather than claims, and untested cases remain unknown.

### Required before public release

- Real, permission-cleared statutes and judgments with reviewable Gazette
  notifications and current amendment status.
- At least 100 **independently lawyer-reviewed** cases in a separate
  `kind: legal` dataset, with negative outcomes and province-specific cases.
  The 120 synthetic cases **do not** count toward this target.
- Human review of answer entailment, current legal effect, and citations.
- Privacy review, account isolation, retention policies and external security
  assessment.
