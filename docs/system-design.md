# Pakistani Lawyer AI — System Design

**Status:** Draft 1.0 · October 2026 · Companion to [PRD](PRD.md) and [TRD](TRD.md).
Notation: solid = implemented today; **(new)** = to be built for the web app.

## 1. System Context

```text
 ┌──────────────┐   HTTPS    ┌───────────────────────────┐
 │  Browser     │ ─────────▶ │  Web app (Next.js)  (new) │
 │  (user)      │ ◀───────── │  SSR pages + static assets│
 └──────────────┘            └─────────────┬─────────────┘
                                           │ JSON / SSE (cookie session)
 ┌──────────────┐  X-API-Key   ┌───────────▼─────────────┐      ┌────────────────────┐
 │ Integrators  │ ───────────▶ │  FastAPI service        │ ───▶ │ LLM provider       │
 └──────────────┘              │  (auth, rate limit,     │      │ local | Anthropic  │
                               │   routes)               │      │ | OpenAI           │
                               │        │                │      └────────────────────┘
                               │  legal_core (domain)    │
                               └───┬───────────────┬─────┘
                                   │               │
                       ┌───────────▼───┐    ┌──────▼──────────────┐
                       │ Search index  │    │ App DB (new)        │
                       │ data/index    │    │ PostgreSQL: users,  │
                       │ vectors+BM25  │    │ sessions, history…  │
                       └───────▲───────┘    └─────────────────────┘
                               │ ingest (offline job)
                       ┌───────┴───────┐
                       │ data/corpus   │  statutes + sources.json (provenance)
                       └───────────────┘
```

## 2. Deployment View

Single-node first (docker-compose):

| Service | Image | Notes |
|---|---|---|
| `web` (new) | Node, Next.js | Talks to `api` over the internal network; the only public entry point besides TLS termination |
| `api` | `python:3.11-slim`, non-root | Index built at image build time; `/health` healthcheck; `./data` volume |
| `db` (new) | PostgreSQL | App data only; the legal corpus index stays file-based until the pgvector/Qdrant decision (TRD §9) |
| reverse proxy (new) | Caddy/nginx | TLS, routes `/` → web, `/api` → api; set `TRUST_PROXY=true` only here |

Scale-out path: stateless `api` replicas behind the proxy require (a) Redis-backed rate limiting and sessions instead of in-process memory, and (b) a shared search index (pgvector/Qdrant/OpenSearch) instead of a local directory.

## 3. Key Flows

### 3.1 Ask a question (streamed)

```text
Browser            Next.js            FastAPI                 legal_core                LLM
  │ submit question   │                  │                         │                     │
  │──────────────────▶│ POST /ask/stream │                         │                     │
  │                   │─────────────────▶│ session/key check       │                     │
  │                   │                  │ rate-limit check        │                     │
  │                   │                  │── ask(question) ───────▶│ analyze intent+juris│
  │                   │                  │                         │ hybrid retrieve     │
  │                   │                  │                         │ min-score gate ──┐  │
  │                   │                  │                         │                  │ none above → status=no_relevant_source
  │                   │                  │                         │ build cited prompt  │
  │                   │                  │                         │────────────────────▶│ generate
  │                   │                  │                         │◀────────────────────│
  │                   │                  │                         │ verify citations ─┐ │
  │                   │                  │                         │                   │ fail → status=citation_verification_failed
  │                   │                  │◀─ QAResult ─────────────│                     │
  │◀── SSE: status ───│◀─────────────────│                         │                     │
  │◀── SSE: delta × n │  (only if answered; 8-word chunks)         │                     │
  │◀── SSE: citations │                                            │                     │
  │◀── SSE: done (disclaimer)                                      │                     │
```

Invariant: text reaches the browser only after verification. The UI renders three terminal states — `answered`, `no_relevant_source` (offer corpus search), `citation_verification_failed` (explain, offer search).

### 3.2 Search with filters

`GET /api/v1/search` → `CorpusSearchService` → hybrid retriever → `RetrievalFilters` (AND of jurisdiction, court, document type, citation, date range) → ranked results with citation, excerpt, `authority_score`/`authority_reason`, metadata. No LLM involved, so it works when generation is down.

### 3.3 Open a source from a citation (new)

Citation marker click → `GET /api/v1/sources/{chunk_id}` → full chunk text, statute structure (Act → section), provenance from `sources.json` (publisher, licence, trust level). The viewer is the verifiability surface promised in PRD §4.2.

### 3.4 Sign-in (new)

`POST /auth/login` → verify Argon2id hash → create server-side session → `Set-Cookie` (HttpOnly, Secure, SameSite=Lax) → subsequent requests authenticated by cookie; state-changing requests carry a CSRF token. Logout revokes the session row.

### 3.5 Ingestion (offline)

```text
data/corpus/*.txt + sources.json
  → parsing → statute_structure (Act/Section) → legal-aware chunking
  → metadata (title, type, jurisdiction, structure) → content-hash chunk IDs
  → embed (only unseen chunks) → vector store + BM25 corpus → data/index
```

Run at image build or via `python -m legal_core.ingest`. Idempotent. Untraceable documents (no `sources.json` entry) must not enter the authoritative corpus (PRD §15).

## 4. Domain Pipeline (inside `legal_core`)

```text
question → analyze_query (intent, jurisdiction)
         → HybridRetriever (BM25 ⊕ vector, filters, authority weighting)
         → min-score gate (RETRIEVAL_MIN_SCORE)
         → AnswerGenerator (prompt = question + numbered sources) → LLMProvider
         → citation parse + verify against retrieved chunks
         → QAResult(status, message, answer, citations, verification, disclaimer)
```

Provider layer is swappable via `LLM_PROVIDER`; the default `local` extractive provider returns source text so everything runs offline.

## 5. Trust and Security Boundaries

```text
[Internet] ─TLS─ [Proxy] ─ [Web] ─ [API] ─ [legal_core] ─ [Index / DB]
                                      └──────────────────▶ [LLM provider]  (outbound; question + retrieved text leave the platform)
```

- Browser is untrusted: validates nothing it is told; all authorization is server-side. No API key or provider key ever ships to the browser.
- Untrusted content: user questions, LLM output, and (later) uploaded documents. Mitigations: length limits, escaped rendering, retrieved text delimited in the prompt, verification before display.
- The LLM provider receives user questions and retrieved excerpts; this must be disclosed (PRD §31) and is a deployment decision (TRD §9.3).
- Secrets: environment variables only; `.env` is git-ignored; the API refuses to start without `PAKISTANI_LAWYER_API_KEY`.

## 6. Failure Modes

| Failure | Behaviour |
|---|---|
| Empty or below-threshold retrieval | `no_relevant_source`; UI links to corpus search |
| Citation verification fails | Answer withheld; `citation_verification_failed` |
| LLM provider error/timeout | Retries then 502 "Answer generation failed"; search remains available |
| Index missing/empty | Retrieval returns empty → no-source state, not a crash |
| Rate limit exceeded | 429 + `Retry-After: 60`; UI shows wait message |
| Embedding provider mismatch with index | Explicit dimension-mismatch error; rebuild the index |
| DB down (new) | Anonymous/API-key Q&A and search keep working; sign-in and history degrade |

## 7. Known Risks

1. **Retrieval quality:** evaluation precision@3 and hit rate@3 are 0.000 with the default hashing embedder; the UI must not launch as a "research tool" until this is addressed.
2. **Relevance gate:** an off-topic query was observed to return `answered`; the min-score threshold is not sufficient with hashing embeddings.
3. **Streaming is post-hoc:** SSE currently replays a completed, verified answer; latency to first token equals full pipeline latency.
4. **In-process state:** rate limiter and (new) sessions are per-process; multi-replica deployment needs Redis.
5. **Tiny, fictional corpus:** the seed corpus is illustrative sample text, not real Pakistani law; the UI must say so until real Level-A sources are ingested.

## 8. Evolution

| Phase | System change |
|---|---|
| Web MVP | Add `web`, `db`, proxy; session auth; `/sources/{id}`; real embedder; fixed relevance gate |
| Phase 2 | Statute/case browsers, history/bookmarks, real public-source ingestion adapters, corpus status page |
| Phase 3 | Upload service (object storage, malware scan, OCR worker via queue), per-user private indexes isolated from the public corpus |
| Phase 4+ | Urdu/Roman Urdu retrieval, citation graph, pgvector/Qdrant/OpenSearch, organization tenancy |
