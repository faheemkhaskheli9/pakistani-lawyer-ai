# Pakistani Lawyer AI — Technical Requirements Document

**Status:** Draft 1.0 · October 2026
**Scope:** the web application (browser UI + backend API) described in [PRD](PRD.md).
**See also:** [System Design](system-design.md) for diagrams and data flow.

Legend: **[Exists]** implemented in `src/legal_core` today · **[New]** required for the web app · **[Later]** phase 2+.

## 1. Goals and Constraints

- Deliver the PRD as a web app: Next.js frontend → FastAPI backend → `legal_core` domain layer.
- Retain the retrieval-first, citation-first design: no answer is sent to the browser unless citation verification passed (the SSE endpoint already enforces this).
- Run fully offline for development and CI (hashing embedder, local extractive LLM, on-disk index).
- Keep legal logic in `legal_core`; API routes stay thin.
- Constraints: small public-domain corpus today; single-node deployment first; no paid vector DB required.

## 2. Technology Choices

| Layer | Choice | Status |
|---|---|---|
| Frontend | Next.js (App Router), TypeScript, Tailwind, shadcn/ui | [New] |
| API | Python, FastAPI, Pydantic, Uvicorn | [Exists] |
| Domain | `legal_core` package | [Exists] |
| Lexical search | In-process BM25 (`hybrid_retrieval.BM25Index`) | [Exists] |
| Vector search | numpy store persisted to `data/index`; embeddings `hashing` (default) or `sentence-transformers` | [Exists] |
| LLM | Provider interface: `local` (extractive), `anthropic`, `openai`, `local-openai` | [Exists] |
| App database | PostgreSQL + SQLAlchemy + Alembic (users, sessions, history, bookmarks) | [New] |
| Production search | PostgreSQL + pgvector, or Qdrant / OpenSearch, behind the existing retriever interface | [Later] |
| Cache / queue | Redis (rate limiting, sessions); Celery/RQ for ingestion and uploads | [Later] |
| Packaging | Docker, docker-compose (`api` today; add `web`, `db`) | [Exists]/[New] |
| CI | GitHub Actions `ci.yml` (pytest); add frontend lint/typecheck/build and Playwright | [Exists]/[New] |

## 3. Components

### 3.1 Backend (`legal_core`) [Exists]

| Module | Responsibility |
|---|---|
| `ingest`, `parsing`, `chunking`, `statute_structure`, `metadata`, `chunk_store`, `embedding_pipeline` | Idempotent ingestion; legal-aware chunking by section/article; per-chunk provenance |
| `embeddings`, `vector_store` | Embedder adapters and persisted vector index |
| `retrieval`, `hybrid_retrieval`, `filters`, `authority` | Top-k retrieval, BM25 + vector merge, metadata filters, authority-aware ranking |
| `query_analysis` | Intent and jurisdiction detection |
| `service` (`QuestionAnsweringService`) | Retrieve → min-score gate → generate → verify citations → `QAResult` |
| `generation`, `llm`, `qa` | Prompt with cited sources, provider calls with retries and error mapping, `LegalAnswer` |
| `citation`, `citation_parser`, `citation_verification` | Citation objects, reporter-citation parsing, verification report |
| `corpus_search` | No-LLM corpus search |
| `safeguards` | Legal-information disclaimer |
| `security` | API-key check (constant time), sliding-window rate limiter, client identity / proxy trust |
| `evaluation` | Precision@k / hit rate, citation spot-checks |

### 3.2 API layer [Exists, extend]

Existing endpoints (`src/legal_core/api.py`):

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Liveness, unauthenticated |
| POST | `/api/v1/ask` | Body `{question}` (1–5000 chars). Returns `QAResult` JSON |
| POST | `/api/v1/ask/stream` | SSE: `status`, `delta`*, `citations`, `done`. Only emitted after verification |
| GET | `/api/v1/search` | `q`, `top_k` (1–100), `jurisdiction`, `court`, `document_type`, `case_citation`, `date_from`, `date_to` |

`QAResult.status` ∈ `answered`, `no_relevant_source`, `citation_verification_failed`. Errors: 400 invalid input, 401 bad key, 429 rate limit (`Retry-After: 60`), 502 generation failure.

[New] endpoints required for the web app:

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/auth/register`, `/login`, `/logout` | Email/password (OAuth providers later) |
| GET | `/api/v1/me` | Current user and role |
| GET | `/api/v1/sources/{chunk_id}` | Source viewer: full chunk text + metadata + provenance |
| GET | `/api/v1/history`, POST `/api/v1/bookmarks`, DELETE `/api/v1/bookmarks/{id}` | History and saved items [Phase 2] |
| GET | `/api/v1/statutes`, `/statutes/{id}`, `/cases`, `/cases/{id}` | Browsers [Phase 2] |
| GET | `/api/v1/corpus/status` | Corpus coverage and ingestion state (admin) [Phase 2] |

### 3.3 Web frontend [New]

- Pages per PRD §38.2; MVP = Ask, Answer view, Search, Source viewer, Sign-in.
- Typed client generated from the FastAPI OpenAPI schema (`/openapi.json`) so request/response types cannot drift.
- Streaming: `fetch` with a `ReadableStream` reader on `POST /api/v1/ask/stream` (EventSource cannot POST); parse `status`/`delta`/`citations`/`done` frames. Render deltas incrementally, show citations on `citations`, show the disclaimer on `done`.
- Rendering rules: answer text is rendered as plain text/escaped markdown — never raw HTML. Source excerpts are quoted blocks visually distinct from AI explanation.
- State: server state via TanStack Query; no secrets in client code or `localStorage`.

## 4. Data Model

Application database (PostgreSQL) [New]:

```text
users(id, email, password_hash, role, created_at, disabled_at)
sessions(id, user_id, token_hash, expires_at, created_at, user_agent)
queries(id, user_id, question, status, created_at, latency_ms)        -- no answer text unless user saves it
bookmarks(id, user_id, chunk_id, note, created_at)
audit_events(id, user_id, action, subject, created_at, ip)
```

Roles: `public`, `registered`, `lawyer`, `org_member`, `org_admin`, `platform_admin` (PRD §28); only `registered` and `platform_admin` are needed for the MVP.

Corpus data (file-based today) [Exists]: `data/corpus/*.txt` + `sources.json` (provenance/licence per document) → `data/index` (chunks, vectors, metadata including `title`, `document_type`, `jurisdiction`, `statute_structure`, `authority_score`, `authority_reason`). Chunk IDs are SHA-256 content hashes (stable across re-ingestion).

## 5. Functional Requirements (technical)

| ID | Requirement |
|---|---|
| F1 | `/ask` and `/ask/stream` shall return `no_relevant_source` when no chunk meets `RETRIEVAL_MIN_SCORE`, with no generated text |
| F2 | A response with `status=answered` shall carry at least one citation, each resolving to a retrieved chunk |
| F3 | If citation verification fails, the answer text shall not be returned or streamed |
| F4 | Every answer response shall include the legal-information disclaimer |
| F5 | Search shall support all filters listed in §3.2 and combine them with AND semantics |
| F6 | Ingestion shall be idempotent: re-running on an unchanged corpus embeds nothing new |
| F7 | The web UI shall display the no-source state, verification status of each citation, and disclaimer for every answer |
| F8 | Unauthenticated browsers shall be limited to a small anonymous quota or none (decision in §9) |

## 6. Authentication and Authorization

Today: one shared secret in `X-API-Key` (`PAKISTANI_LAWYER_API_KEY`, required, no default), compared in constant time. This is **unsuitable for browsers** — embedding the key in a frontend bundle would publish it.

Web app design [New]:

- Browser users authenticate with email/password (Argon2id hashes) and receive an opaque session token in an `HttpOnly; Secure; SameSite=Lax` cookie; sessions stored server-side with expiry and revocation.
- CSRF: `SameSite=Lax` plus a double-submit CSRF token on state-changing requests; CORS restricted to the web origin with credentials.
- The shared `X-API-Key` is retained for server-to-server integrators; the API guard accepts either a valid session or a valid key.
- Rate limiting keyed by user ID when authenticated, otherwise client IP (the existing `client_identity`; `TRUST_PROXY=true` only behind a trusted proxy). Move limiter state to Redis when running more than one API process (the current limiter is in-process memory).
- Authorization: role checks in a FastAPI dependency; private user data always filtered by `user_id`.

## 7. Non-Functional Requirements

| Area | Target |
|---|---|
| Latency | Search p95 < 2 s; time-to-first-token for streamed answers < 3 s; full answer < 10 s typical (PRD §44). Note the current `/ask/stream` runs the full QA pipeline first and then streams the finished text, so "streaming" does not yet reduce time-to-first-token — true token streaming from the LLM is a [Later] improvement that must keep post-verification gating |
| Availability | > 99.5 % (product target, not a launch guarantee) |
| Security | TLS everywhere; secrets via environment/secret manager; input length limits (5000 chars on questions); dependency scanning in CI; non-root container user [Exists] |
| Privacy | No logging of question text beyond the `queries` table when the user has history enabled; delete-my-data endpoint; document uploads (Phase 3) isolated per user and never merged into the public corpus |
| Accessibility | WCAG 2.1 AA |
| Browser support | Last 2 versions of Chrome, Edge, Firefox, Safari |
| Observability | Structured logs with request ID; metrics for retrieval/generation latency, status counts (`answered` / `no_relevant_source` / `citation_verification_failed`), 4xx/5xx, rate-limit hits; no document contents in logs |

## 8. Quality, Testing, and Evaluation

- Backend: pytest (`tests/`, ~30 modules including API, retrieval quality, citation verification, security) [Exists].
- Retrieval evaluation: `python -m legal_core.evaluation` over `data/evaluation/retrieval_queries.json`. **Current baseline is precision@3 = 0.000 and hit rate@3 = 0.000** with the default hashing embedder; the web launch gate is a documented, reproducible improvement on this baseline with a real embedder (target to be set once the labelled set is expanded and expert-reviewed).
- Known defect to resolve before launch: off-topic questions can pass the relevance gate and return `answered` (observed with a physics query). Add negative cases to the evaluation set and tune the gate against them.
- Frontend: typecheck + lint + component tests; Playwright end-to-end covering ask (answered, no-source, verification-failed), search with filters, source viewer, auth, and 429 handling.
- CI gates: backend tests, frontend build/typecheck, e2e against the docker-compose stack with the offline configuration.

## 9. Open Decisions

1. Anonymous access: allow a small free quota without sign-in, or require an account for all asks?
2. Default embedder for the web deployment (hashing is offline-friendly but scores poorly; `sentence-transformers` adds a model download and CPU/RAM cost).
3. Hosted LLM provider and data-handling terms for user questions (PRD §31 requires documenting whether prompts leave the platform).
4. Whether the frontend is served as a separate container or statically exported behind the API's reverse proxy.
5. When to migrate from the on-disk index to pgvector/Qdrant (trigger: corpus size or multi-node deployment).

## 10. Milestones

1. **M0 — Quality gate:** fix retrieval baseline and no-source gating; expand evaluation set.
2. **M1 — Auth + sessions:** user accounts, cookie sessions, CORS/CSRF, per-user rate limits.
3. **M2 — Web MVP:** Ask, Answer view (streaming), Search, Source viewer, Sign-in; `/sources/{id}` endpoint; compose stack; Playwright e2e.
4. **M3 — Phase 2:** statute/case browsers, history and bookmarks, corpus status, real public-source ingestion adapters.
5. **M4 — Phase 3:** uploads, document Q&A, workspaces (separate storage, malware scanning, retention controls).
