# Verified legal corpus onboarding — Phase 1 of issue #28

**Status:** In progress. This phase implements admission controls, not a verified
Pakistani law database. The three bundled `sample_*.txt` files remain fiction.

## Goal and acceptance gates

Build a traceable, legally reusable collection of Pakistani law without
presenting a potentially outdated portal copy as the authoritative Gazette.

1. **Source discovery (implemented):** maintain an official-source directory of
   URLs. Do not scrape court portals, download bulk documents or assume
   publishing equals permission to redistribute.
2. **Rights/provenance review (implemented admission gate):** a human reviewer
   confirms source URL, publisher, statute/judgment type, jurisdiction,
   applicable Gazette, effective/decision date, current status, and
   permission to store and redistribute. Record a rights evidence reference.
3. **Local intake (implemented):** place rights-cleared `.txt` or searchable
   `.pdf` documents directly in the corpus directory, alongside
   `source_registry.json`. Each document is SHA-256 locked. Future changes
   require a new review and new hash.
4. **Fail closed (implemented):** strict ingestion rejects unlisted, altered,
   superseded, unrevised, unlicensed, symlinked and nested documents. `APP_ENV=production`
   also rejects indexes without the admission metadata. Do not ingest or
   publish confidential client files.
5. **Validation (in progress):** human reviewers must compare extracted
   content to official Gazettes, sample citations, amendments, and relevant
   judicial precedent. A registry flag does **not** automate this legal review.
6. **Quality benchmark (remaining):** 100+ expert-labeled questions by
   statute, including explicit negatives, outdated provisions, legal changes,
   jurisdiction boundaries, Urdu/Roman Urdu, and contradictory case law.
7. **Operations (remaining):** secure authenticated access, controlled review
   renewal, monitoring, backups, incident response and lawyer user study.

## Official discovery sources

- Pakistan Code: https://pakistancode.gov.pk/english/ — Ministry of Law and
  Justice. The portal warns the text is under review and advises consulting
  relevant Gazette notifications where in doubt.
- National Assembly downloads: https://www.na.gov.pk/en/downloads.php —
  versioned Constitution and legislative materials, not a reuse licence.
- Sindh Law Department: https://sindhlaws.gov.pk/ — provincial law references.

All of these are **discovery links, not permission grants**. Review the
original publication and terms before using or redistributing any text.

## How to admit a locally reviewed source

1. Obtain an authorized copy and written permission or a documented rights
   basis to store and, where applicable, redistribute it. Record the evidence.
2. Verify the official published copy and the relevant Gazette (statutes) or
   the court's original judgment (cases). Confirm amendments/repeals.
3. Put the file directly under a private corpus directory, e.g.
   `/srv/legal-corpus/contract_act.pdf`. Keep it outside Git.
4. Create `/srv/legal-corpus/source_registry.json` using
   `docs/source_registry.example.json` as the schema; provide a real
   SHA-256 for the exact bytes and a human reviewer/date.
5. Run:

   ```bash
   python -m legal_core.source_registry --corpus /srv/legal-corpus
   python -m legal_core.ingest --corpus /srv/legal-corpus \
      --index-dir /srv/legal-index --admitted-only
   ```

6. Point `INDEX_DIR` to `/srv/legal-index` and restart the API. In production
   the index must contain reviewed, admitted sources only. Check `/ready`.
7. Repeat legal review at least every 90 days. Re-index after every amendment
   or repeal and verify source excerpts with a lawyer.

## Example schema

`docs/source_registry.example.json` is illustrative only: it has a placeholder
hash and intentionally marks permission and legal status as **unapproved**, so
it will not pass admission. Do not edit `source_registry.json` to falsely
mark a law as current or its reproduction as authorized.

The registry checks integrity and recorded approvals; it cannot establish
copyright ownership, jurisdiction-specific legal effect, or factual accuracy.
Production readiness requires an actual lawyer-verified corpus and tests.
