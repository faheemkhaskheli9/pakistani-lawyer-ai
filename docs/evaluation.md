# Evaluation Notes: Pakistani Lawyer AI

## Metrics

- **Retrieval precision@k** — for a small hand-labeled set of
  question -> expected-source-section pairs, the fraction of top-k retrieved
  chunks that match the expected source.
- **Citation accuracy (manual spot check)** — for a sample of generated
  answers, whether every factual claim is actually supported by the cited
  chunk (not just topically related to it).
- **"No relevant source" precision** — for out-of-corpus questions, the
  fraction of responses that correctly say no relevant source was found
  instead of fabricating an answer.

## Corpus licensing note

The seed corpus under `data/corpus/` (issue #3) is original sample
statute-style text authored for this project, not a reproduction of any
real statute — see `README.md` §8 for why. `data/corpus/sources.json`
lists, per document: title, source URL (null for these self-authored
documents), and the basis for including it. No document with an unclear or
restrictive license is included.

## Result Log

_Fill in as phases land — one row per evaluation run._

| Date | Phase | Metric | Value | Notes |
|------|-------|--------|-------|-------|
| _TBD_ | | | | |
