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
