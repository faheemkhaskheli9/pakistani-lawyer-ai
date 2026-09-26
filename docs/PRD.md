# Pakistani Lawyer AI — Product Requirements Document

**Repository:** `faheemkhaskheli9/pakistani-lawyer-ai`  
**Product Type:** Legal Research, Document Intelligence, and AI Assistant Platform  
**Primary Market:** Pakistan  
**Status:** In development  
**Version:** 1.0  
**Date:** September 2026

## 1. Product Overview

Pakistani Lawyer AI is an AI-powered legal research and document intelligence platform focused on Pakistani law.

The platform helps users search statutes, regulations, judgments, legal documents, and other authoritative legal materials using natural language.

The system combines:

- legal document ingestion;
- semantic and keyword search;
- Retrieval-Augmented Generation (RAG);
- citation-grounded AI answers;
- case-law research;
- document analysis;
- legal document summarization;
- multilingual interaction;
- research workspaces;
- citation verification;
- lawyer productivity tools.

The product must prioritize **source verification and traceability** over unrestricted LLM generation.

The AI should not invent laws, judgments, sections, citations, or legal authorities.

The platform provides legal information and research assistance. It is **not a substitute for professional legal advice**.

## 2. Problem

Pakistani legal information is fragmented across federal and provincial government websites, Supreme Court and High Court resources, tribunal and regulator websites, PDF statutes, notifications, amendments, gazettes, and reported judgments.

Users often need to know which law applies, what a section says, whether a law has been amended, what courts have said about an issue, which authorities support or oppose an argument, and where an AI-generated answer came from.

General-purpose LLMs create an additional risk: plausible-looking but nonexistent Pakistani cases or statutory provisions.

Pakistani Lawyer AI addresses this through a **retrieval-first and citation-first architecture**.

## 3. Product Vision

Create a trusted AI research layer for Pakistani law where a user can ask a legal question and receive:

1. a concise explanation;
2. relevant statutes;
3. specific sections;
4. related judgments;
5. source excerpts;
6. links to original sources;
7. evidence-quality indicators;
8. conflicting or alternative authorities where applicable.

The long-term goal is to evolve from a legal Q&A application into a complete **AI legal research workspace**.

## 4. Product Principles

### 4.1 Sources Before Generation
The system should search authoritative material before asking an LLM to formulate an answer.

### 4.2 Every Legal Claim Should Be Verifiable
Users should be able to inspect the supporting source behind important claims.

### 4.3 No Citation Fabrication
The AI must not generate nonexistent cases, statutes, sections, court names, dates, citations, or quotations.

### 4.4 Retrieval Failure Is Better Than Hallucination
If sufficient information cannot be found, the system should clearly state that relevant authoritative material was not found in the available corpus.

### 4.5 Separate Law From AI Interpretation
The UI should distinguish source text, extracted facts, AI explanation, and AI-generated summaries.

### 4.6 Law Is Versioned
The platform must eventually support enactment dates, amendment history, repealed provisions, effective dates, and jurisdiction.

### 4.7 Privacy by Design
Uploaded client or case documents must be treated as confidential information.

## 5. Target Users

Primary users:

- practicing lawyers;
- junior associates;
- law students;
- legal researchers;
- law firms.

Secondary users:

- businesses researching contracts, employment, taxation, corporate compliance, and regulatory obligations;
- members of the public using simplified legal-information mode.

## 6. Core User Journey

```text
User enters legal question
        ↓
Query classification
        ↓
Determine legal domain/jurisdiction
        ↓
Hybrid retrieval
        ↓
Retrieve statutes + cases + regulations
        ↓
Rerank authorities
        ↓
Generate source-grounded answer
        ↓
Verify citations
        ↓
Return structured legal research answer
        ↓
User opens sources or asks follow-up
```

## 7. Core Product Modules

### 7.1 AI Legal Research Assistant

Users can ask legal questions in plain language.

Responses should contain:

- summary;
- relevant law;
- specific sections;
- relevant cases;
- analysis grounded in retrieved materials;
- sources;
- limitations and uncertainty.

### 7.2 Legal Corpus Search

Search should support:

- exact text;
- keyword;
- semantic;
- statute;
- section;
- citation;
- case title;
- judge;
- court;
- year;
- legal topic.

Filters should include court, jurisdiction, year, legal category, and document type.

### 7.3 Statute Explorer

Users should be able to browse legal text hierarchically by act, chapter, section, subsection, explanation, and schedule.

Each provision should eventually show current text, enactment date, jurisdiction, amendments, related cases, related provisions, definitions, and source URL.

### 7.4 Case Law Explorer

Each judgment should have structured metadata including:

- case name;
- court;
- judges;
- case number;
- judgment date;
- reported citation;
- parties;
- legal topics;
- statutes referenced;
- cases cited;
- procedural history;
- disposition;
- source.

AI-generated summaries should clearly identify facts, issues, arguments, holding, reasoning, ratio, observations, and outcome.

### 7.5 Case Similarity Search

Users can search using a case, citation, judgment, fact pattern, or legal issue.

Similarity should consider issues, facts, statutes, legal concepts, and court hierarchy.

### 7.6 Citation Search

Users can search by citation format such as PLD, SCMR, CLC, MLD, and YLR.

Citation parsing should identify reporter, year, court, page, and case.

### 7.7 Citation Verification

Every generated citation should pass through a verification layer.

```text
LLM answer
   ↓
Extract cited authorities
   ↓
Compare against retrieved source metadata
   ↓
Validate citation
   ↓
Display verified citation
```

Unsupported citations should be removed, marked unverified, or trigger answer regeneration.

### 7.8 Legal Document Upload

Users should be able to upload PDF, DOCX, TXT, scanned PDF, and images.

Supported document types include contracts, FIRs, legal notices, petitions, judgments, agreements, and pleadings.

The system should extract text, headings, parties, dates, amounts, clauses, sections, and legal references.

### 7.9 Ask Questions About Uploaded Documents

Answers should cite page, clause, and paragraph where possible.

### 7.10 Legal Document Comparison

Users can compare draft vs executed agreements, old vs new agreements, or amended vs original legal text.

The system should highlight additions, deletions, changed obligations, dates, financial terms, and important clause changes.

### 7.11 Document Summarization

Support short summaries, detailed summaries, executive summaries, and structured case briefs.

### 7.12 Contract Analysis

Future clause extraction should cover termination, indemnity, liability, confidentiality, payment, governing law, arbitration, renewal, penalties, intellectual property, and force majeure.

The system may flag clauses for review but should not present risk labels as definitive legal conclusions.

### 7.13 Legal Research Workspace

Users can create matters containing questions, cases, statutes, uploaded documents, notes, AI conversations, and draft research memos.

## 8. Legal Domains

The system should eventually support constitutional, criminal, civil, family, corporate, company, employment, banking, taxation, property, contract, intellectual property, cybercrime, privacy, consumer, administrative, environmental, telecommunications, securities, and competition law.

Coverage must be measurable rather than implying comprehensive Pakistani-law coverage before it exists.

## 9. Jurisdiction Support

Metadata should support:

- Federal / Pakistan-wide law;
- Punjab;
- Sindh;
- Khyber Pakhtunkhwa;
- Balochistan;
- Islamabad Capital Territory.

Jurisdiction should influence filtering and ranking.

## 10. Multilingual Support

Target languages:

- English;
- Urdu;
- Roman Urdu.

The system should map multilingual queries to relevant legal concepts while preserving original legal terminology where useful.

## 11. Search Architecture

Use hybrid retrieval.

```text
Query
  ↓
Query preprocessing
  ↓
Keyword Search / BM25
           +
Vector Search
  ↓
Candidate Merge
  ↓
Reranker
  ↓
Authority / metadata filters
  ↓
Top legal passages
```

Legal search should not rely on vector similarity alone because exact terminology, citations, section numbers, and case names matter.

## 12. Retrieval Ranking

Ranking should consider:

- semantic similarity;
- keyword relevance;
- court authority;
- jurisdiction match;
- citation match;
- document type;
- recency;
- source quality.

## 13. RAG Architecture

```text
Legal Sources
      ↓
Source validation
      ↓
Parser / OCR
      ↓
Metadata extraction
      ↓
Legal-aware chunking
      ↓
Embeddings
      ↓
Search Index
      ↓
Hybrid Retriever
      ↓
Reranker
      ↓
Context Builder
      ↓
LLM
      ↓
Citation Validator
      ↓
Response
```

## 14. Chunking Strategy

Statutes should be chunked by legal hierarchy such as act, chapter, section, subsection, clause, explanation, and schedule.

Judgments should prefer paragraph, issue, legal reasoning, holding, and order boundaries rather than fixed token chunks only.

Chunk metadata should preserve document hierarchy and provenance.

## 15. Source Management

Each source should record:

- title;
- publisher;
- source URL;
- acquisition date;
- jurisdiction;
- court;
- date;
- document type;
- licensing/provenance;
- checksum;
- version;
- ingestion status.

Untraceable content should not enter the authoritative corpus.

## 16. Source Trust Levels

Suggested source hierarchy:

- **Level A:** official government or court source;
- **Level B:** recognized legal reporting or authenticated source;
- **Level C:** secondary commentary.

AI answers should prioritize authoritative primary material whenever available.

## 17. Query Classification

Classify intent before retrieval:

- statute lookup;
- case search;
- citation lookup;
- legal question;
- document question;
- case summary;
- document comparison;
- legal definition;
- research request.

## 18. AI Response Format

Responses should use structured sections such as:

- Summary;
- Relevant Law;
- Relevant Sections;
- Relevant Cases;
- Explanation;
- Sources;
- Limitations.

Each source should expose document title, court or authority, section/paragraph, source link, and retrieved excerpt.

## 19. Evidence Quality

Avoid unsupported numerical confidence scores.

Prefer evidence indicators such as:

- source coverage;
- number of authoritative sources;
- direct statutory authority available;
- relevant case law available;
- conflicting authorities detected.

## 20. Lawyer Mode

Professional research features:

- advanced filters;
- citation search;
- Boolean search;
- case history;
- related cases;
- statute relationships;
- research notes;
- export citations;
- source excerpts;
- matter workspaces.

## 21. Public Mode

Provide a simplified legal-information interface with plain-language explanations, important laws, source links, and a visible legal-information disclaimer.

## 22. Research Memo Generator

Generate structured research memos containing:

- Question Presented;
- Short Answer;
- Relevant Law;
- Authorities;
- Analysis;
- Counterarguments;
- Conclusion;
- Source List.

Substantive propositions should link back to selected authorities.

## 23. Timeline Extraction

Extract important dates and events from cases or uploaded documents to create matter timelines.

## 24. Entity Extraction

Extract entities such as:

- people;
- companies;
- courts;
- judges;
- statutes;
- sections;
- cases;
- agencies;
- dates;
- locations;
- monetary amounts.

## 25. Legal Knowledge Graph

Long-term model:

```text
Case
 ├── cites → Case
 ├── interprets → Section
 ├── applies → Statute
 ├── decided_by → Court
 └── involves → Legal Issue
```

This can support citation networks, related-case discovery, legal authority exploration, and statute-to-case navigation.

## 26. Search History and Saved Research

Authenticated users should be able to store recent searches, recent cases, bookmarks, saved statutes, excerpts, notes, conversations, and research items.

## 27. Export

Support export to:

- PDF;
- DOCX;
- Markdown;
- plain text.

Exports should preserve legal citations and source links.

## 28. Authentication and Roles

Initial authentication options:

- email/password;
- Google;
- GitHub.

Roles:

- Public User;
- Registered User;
- Lawyer;
- Organization Member;
- Organization Admin;
- Platform Admin.

## 29. Organization Workspaces

Future law-firm support should include members, matters, documents, research, conversations, permissions, shared notes, and tenant isolation.

## 30. Security Requirements

Required safeguards include:

- TLS;
- encrypted storage;
- secure object storage;
- signed file URLs;
- role-based access control;
- organization isolation;
- audit logs;
- secure secret management;
- input validation;
- malware scanning;
- file size limits;
- API rate limiting;
- document deletion;
- configurable retention.

Private documents must never automatically become part of the public legal corpus.

## 31. Privacy Requirements

Users should be able to delete uploaded documents, conversations, and workspaces and export their data.

The application should document retention, provider usage, whether prompts/documents leave the platform, and logging behavior.

## 32. Safety Requirements

The system must not:

- invent laws;
- fabricate cases;
- fabricate citations;
- claim certainty where sources are missing;
- imply that generated output guarantees a legal outcome.

Every answer view should display an appropriate legal-information notice.

## 33. Hallucination Guardrails

```text
Generated claim
       ↓
Citation attached?
       ↓
Citation exists?
       ↓
Source supports claim?
       ↓
Display claim
```

Unsupported legal claims should be removed, regenerated, or explicitly marked unsupported.

## 34. LLM Architecture

Use a provider abstraction so OpenAI, Azure OpenAI, Anthropic, Gemini, local models, or future providers can be swapped without changing core legal logic.

## 35. Embedding Architecture

Support local and hosted embedding adapters.

Development should continue to work offline where possible.

## 36. Search and Storage

Local development may continue with FAISS or Chroma.

Production options include:

- PostgreSQL + pgvector;
- Qdrant;
- OpenSearch / Elasticsearch;
- Weaviate.

A dedicated lexical search engine plus vector search may eventually be preferable for legal retrieval.

## 37. Backend Stack

Recommended:

- Python;
- FastAPI;
- Pydantic;
- SQLAlchemy;
- PostgreSQL;
- pgvector;
- Redis;
- Celery or RQ;
- OpenSearch / Elasticsearch when required.

The existing `legal_core` package should remain the domain layer rather than placing legal logic directly inside API routes.

## 38. Frontend

Recommended:

- Next.js;
- TypeScript;
- Tailwind CSS;
- shadcn/ui.

Primary interfaces:

- AI search;
- research results;
- source viewer;
- statute browser;
- case browser;
- document viewer;
- research workspace.

## 39. API

Potential endpoints:

```text
POST /api/v1/ask
GET  /api/v1/search
GET  /api/v1/statutes
GET  /api/v1/statutes/{id}
GET  /api/v1/cases
GET  /api/v1/cases/{id}

POST /api/v1/documents
GET  /api/v1/documents/{id}
POST /api/v1/documents/{id}/ask
POST /api/v1/documents/compare

POST /api/v1/research
GET  /api/v1/research/{id}
```

## 40. Core Data Models

Core entities:

- User;
- Organization;
- Matter;
- Document;
- DocumentVersion;
- LegalSource;
- Statute;
- Section;
- Case;
- Court;
- Judge;
- Citation;
- LegalTopic;
- Chunk;
- Embedding;
- Conversation;
- Message;
- ResearchItem;
- Bookmark;
- Note;
- AuditEvent.

## 41. Evaluation

Retrieval metrics:

- Precision@K;
- Recall@K;
- MRR;
- NDCG.

Citation metrics:

- citation existence;
- citation correctness;
- citation relevance;
- citation-to-claim consistency.

Generation evaluation should track groundedness, factual consistency, unsupported claims, completeness, and source coverage.

## 42. Evaluation Dataset

Create a curated benchmark containing:

- question;
- expected statute;
- expected section;
- expected cases;
- expected jurisdiction;
- expected source type;
- reference answer.

Legal-expert review should be included before treating this benchmark as authoritative.

## 43. Observability

Production should record:

- query latency;
- retrieval latency;
- generation latency;
- token usage;
- retrieval score;
- selected sources;
- citation validation failures;
- ingestion failures.

Do not log private document contents unnecessarily.

## 44. Performance Targets

Initial targets:

- Search: < 2 seconds;
- AI answer: < 10 seconds typical;
- Document search: < 3 seconds;
- API availability: > 99.5%.

These are product targets rather than launch guarantees.

## 45. MVP

The first production-style MVP should include:

1. Legal document ingestion
2. Structured metadata
3. Hybrid legal search
4. RAG question answering
5. Source-grounded responses
6. Inline citations
7. Citation verification
8. No-source fallback
9. Corpus browser
10. Basic API
11. Authentication
12. Rate limiting
13. Basic web interface
14. Evaluation suite

## 46. Roadmap

### Phase 1 — Complete Current Core
- retrieval-quality tests;
- grounded answer generation;
- inline citations;
- relevance threshold;
- no-source fallback;
- citation verification;
- FastAPI endpoints.

### Phase 2 — Legal Search Product
- public-source ingestion adapters;
- statute browser;
- case explorer;
- advanced search filters;
- case summaries;
- user accounts;
- saved searches;
- bookmarks.

### Phase 3 — Document Intelligence
- document uploads;
- PDF/OCR processing;
- document Q&A;
- clause extraction;
- document comparison;
- research workspaces;
- research memo generation.

### Phase 4 — Advanced Legal Intelligence
- Urdu and Roman Urdu;
- citation graph;
- similar-case discovery;
- amendment tracking;
- legal knowledge graph;
- organization workspaces.

### Phase 5 — Professional Platform
- law-firm accounts;
- team collaboration;
- shared research;
- matter management;
- audit logs;
- organization knowledge bases;
- private RAG collections;
- API access.

## 47. Existing Repository Alignment

The repository already includes or is working toward:

- `legal_core`;
- legal chunking;
- embeddings;
- ingestion;
- retrieval;
- vector storage;
- source/citation metadata;
- retrieval tests;
- seed corpus;
- offline execution;
- configurable top-K retrieval.

Existing open roadmap work includes grounded answer generation, inline citations, no-source fallback, corpus browsing, disclaimer enforcement, authentication, rate limiting, Docker, and an evaluation harness.

These components should be retained and extended rather than rebuilt.

## 48. Immediate Engineering Priorities

Recommended implementation order:

1. Finish existing retrieval-quality tests.
2. Implement provider-independent grounded answer generation.
3. Add citation objects to API responses.
4. Implement configurable relevance threshold.
5. Add claim/citation verification.
6. Create FastAPI endpoints.
7. Create corpus search endpoint.
8. Add authentication and rate limits.
9. Create the initial web UI.
10. Add hybrid lexical + vector search.
11. Improve legal-aware metadata.
12. Introduce real public-source ingestion adapters.
13. Build a reproducible evaluation suite.

## 49. Target Architecture

Evolve from:

```text
Question
↓
Vector Search
↓
LLM
```

to:

```text
Question
↓
Intent Detection
↓
Jurisdiction Detection
↓
Query Expansion
↓
Keyword + Vector Retrieval
↓
Metadata Filtering
↓
Reranking
↓
Legal Authority Selection
↓
LLM
↓
Citation Validation
↓
Claim Validation
↓
Answer
```

## 50. Non-Goals

The product should not initially attempt to:

- replace lawyers;
- predict court outcomes;
- automatically determine guilt;
- make binding legal conclusions;
- file court documents automatically;
- represent users in proceedings;
- claim comprehensive Pakistani-law coverage before such coverage exists.

## 51. Success Metrics

Search:
- successful search rate;
- source click-through;
- zero-result rate;
- retrieval precision.

AI:
- citation correctness;
- grounded-answer rate;
- unsupported-claim rate;
- no-source detection accuracy.

Product:
- weekly active users;
- searches per user;
- saved research;
- document uploads;
- return rate.

## 52. Long-Term Direction

Pakistani Lawyer AI can evolve into:

```text
Pakistan Legal Search Engine
        +
AI Legal Research Assistant
        +
Case Law Intelligence
        +
Statute Intelligence
        +
Legal Document AI
        +
Law Firm Research Workspace
```

The strongest long-term asset is the combination of a structured legal corpus, legal metadata, citation graph, hybrid retrieval, legal evaluation dataset, citation verification, and lawyer workflow tools.

## 53. Product Definition

> **An AI-powered Pakistani legal research and document intelligence platform that helps users find, understand, compare, and verify legal information through authoritative sources and citation-grounded AI.**

Central design rule:

> **No legal answer without evidence.**
