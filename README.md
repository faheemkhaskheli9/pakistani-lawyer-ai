# Pakistani Lawyer AI

> Legal AI / Document-AI portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

Pakistani statutes, procedural codes, and reported case law are published
across scattered government and court websites with no unified full-text
search or citation-grounded Q&A. A person or junior researcher trying to
answer "what does the law say about X" has to know which act to open and
manually search a PDF. This project builds a retrieval-grounded assistant
that answers legal-information questions over a small, curated corpus of
public-domain Pakistani legal text and always cites the specific
section/judgment it drew from.

**This is a legal-information demo, not legal advice, and not a lawyer
replacement.** Every answer is expected to carry a visible disclaimer and a
citation back to source text; the project does not attempt to predict case
outcomes or give jurisdiction-specific counsel.

## 2. Architecture

```text
Corpus (public statutes/judgments, .txt/.pdf)
    -> Ingestion (chunk + embed, source citation kept per chunk)
    -> Vector store
User question -> Retrieve top-k chunks -> LLM answer grounded in retrieved text
    -> Answer + cited section/judgment references
```

A `legal_core` layer owns ingestion (chunking, embedding, idempotent
re-ingestion) and retrieval; a thin API/UI layer handles the question/answer
loop and renders citations next to each answer.

## 3. Technology Stack

- Python, FastAPI (API layer)
- sentence-transformers for embeddings, a local vector store (FAISS or
  Chroma) — no paid vector DB required
- LLM behind a swappable provider interface (see `multi-llm-router` in this
  portfolio for the general provider-swap pattern) — a fake/local backend by
  default so the project runs with no API key
- Pytest for tests

## 4. Feature List

- Ingest a small curated corpus of public-domain Pakistani statutes
  (e.g. sections of a public act obtained from a government open-data
  source) and publicly redistributable case-law summaries
- Ask a legal-information question in plain language and get a grounded
  answer with inline citations back to the source section/judgment
- Browse/search the ingested corpus directly (no LLM required) as a
  fallback when a grounded answer isn't available
- Every answer displays a "not legal advice" disclaimer and a confidence /
  "no relevant source found" signal when retrieval comes up empty

## 5. Implementation Plan

1. Phase 1: `legal_core` shared app — corpus ingestion (chunking + embedding,
   idempotent re-ingestion, per-chunk source citation) and a small seed
   corpus of public-domain sample statute text, project skeleton
2. Phase 2: Retrieval layer — top-k retrieval over the vector store, a
   retrieval-quality smoke test against the seed corpus
3. Phase 3: Question-answering endpoint — LLM-generated answer grounded in
   retrieved chunks, citations rendered alongside the answer, "no relevant
   source found" fallback path
4. Phase 4: Corpus browse/search UI (no-LLM fallback), disclaimer UI, basic
   auth/rate limiting on the QA endpoint
5. Phase 5: Docker/compose, CI, docs, evaluation harness (retrieval
   precision@k + citation-accuracy spot checks)

## Task Tracking

Work will be broken into phase-tagged user stories tracked as GitHub Issues,
not in this file. Implement Phase 1 issues first (later phases depend on it).
When you start one, add label `status:in-progress`. When you finish, close it
referencing the commit (e.g. `git commit -m "... Closes #4"`) and push.

## 6. Repository Structure

```text
pakistani-lawyer-ai/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd pakistani-lawyer-ai
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

The checked-in seed corpus (`data/corpus/`) is **original sample text
authored for this project** in a generic, clearly-fictional statute style
("Sample Civil Code", "Sample Criminal Procedure Code", "Sample Evidence
Act") — not a reproduction of, or derived from, any specific real
Pakistani statute. This avoids any copyright question over real statute
text while still exercising the full ingestion/chunking/citation/embedding
pipeline end-to-end. `data/corpus/sources.json` records, per document, the
source/license basis for including it. No scraped, copyrighted, or
client-identifiable legal documents are used. The seed corpus is
intentionally small (a demo, not a production legal database) — see
`docs/evaluation.md` for exactly what's included and its licensing note.

## 9. Training / Execution

```bash
# Once Phase 1 lands:
python -m legal_core.ingest --corpus data/corpus
uvicorn src.api:app --reload   # once Phase 3 lands
```

**From VSCode:** open the repo root as the workspace, then Run and Debug ->
"legal_core: ingest seed corpus" (`.vscode/launch.json`) runs the same
command under `debugpy` with `cwd` set to the repo root and `PYTHONPATH`
pointing at `src/`, so breakpoints in `src/legal_core/*.py` are hit without
an editable install. A second config, "Python: Debug Tests (pytest)", runs
the test suite the same way.

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see
`docs/evaluation.md`): retrieval precision@k against a small labeled query
set, and manual citation-accuracy spot checks.

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics
tables, and sample outputs go here._

## 12. API

_Once Phase 3 lands: document the QA endpoint here (or link to
auto-generated OpenAPI docs at `/docs`)._

## 13. Docker

```bash
docker build -t pakistani-lawyer-ai .
docker run -p 8000:8000 pakistani-lawyer-ai
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio
  purposes. It is not legal advice and must not be used as a substitute for
  a licensed lawyer.
- The corpus is a small public-domain sample, not a comprehensive or
  up-to-date legal database — answers are only as complete as the seed
  corpus.
- Scaffold stage: no code has been implemented yet — see §5 Implementation
  Plan.

## 16. Future Work

- Expand the corpus (still public-domain only) to cover more statutes.
- Add a citation-verification pass that checks a generated answer's claims
  against the retrieved text before showing it.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the
kind of production systems I have worked on professionally**. It contains no
employer or client source code, prompts, datasets, credentials, architecture
diagrams, or business logic. All code, data, and documentation here are
original or built on publicly available datasets and open-source tools.
This project is not legal advice and is not affiliated with any law firm,
court, or government body.

---
_Last updated: 2026-09-12_
