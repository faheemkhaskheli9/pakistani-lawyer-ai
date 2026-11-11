"""Audited, local-only intake of Pakistani legal sources.

A registry provides machine-checkable provenance and human attestation.
It does NOT prove a law is in force or that redistribution rights exist;
those claims must be reviewed by a qualified person before admission.
No web scraping or downloading occurs in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from .parsing import SUPPORTED_EXTENSIONS

REGISTRY_FILE = "source_registry.json"
MAX_REVIEW_AGE_DAYS = 90
OFFICIAL_HOSTS = frozenset({
    "pakistancode.gov.pk",
    "na.gov.pk",
    "senate.gov.pk",
    "pakistan.gov.pk",
    "sindhlaws.gov.pk",
    "punjablaws.gov.pk",
    "punjabcode.punjab.gov.pk",
    "kpcode.kp.gov.pk",
    "balochistan.gov.pk",
    "supremecourt.gov.pk",
    "lhc.gov.pk",
    "sindhhighcourt.gov.pk",
    "peshawarhighcourt.gov.pk",
    "bhc.gov.pk",
    "ihc.gov.pk",
})
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
DOC_TYPES = frozenset({"statute", "judgment"})


class SourceRegistryError(ValueError):
    """A document failed admission; no corpus write should occur."""


def _required(entry: dict, field: str, filename: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SourceRegistryError(f"{filename}: missing or blank {field}")
    return value.strip()


def _official_url(value: str, field: str, filename: str) -> str:
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        port = parsed.port
    except ValueError as exc:
        raise SourceRegistryError(f"{filename}: malformed {field} URL") from exc
    if (
        parsed.scheme != "https"
        or host not in OFFICIAL_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or not parsed.path
    ):
        raise SourceRegistryError(
            f"{filename}: {field} must be an HTTPS URL on an approved official host"
        )
    return value


def _date(entry: dict, field: str, filename: str, today: date) -> str:
    raw = _required(entry, field, filename)
    try:
        value = date.fromisoformat(raw)
    except ValueError as exc:
        raise SourceRegistryError(f"{filename}: invalid ISO date in {field}") from exc
    if value > today:
        raise SourceRegistryError(f"{filename}: {field} cannot be in the future")
    return value.isoformat()


def validate_registry(
    corpus_dir: str | Path, *, today: date | None = None
) -> dict[str, dict]:
    """Validate a complete local corpus before altering any index.

    All present .txt/.pdf documents must be listed exactly once. SHA-256,
    rights sign-off, official provenance, current status and fresh review
    are required. 'Authorized' and 'current' are explicit *human*
    attestations; this process cannot independently establish either.
    """
    root = Path(corpus_dir)
    manifest = root / REGISTRY_FILE
    if not manifest.is_file():
        raise SourceRegistryError(f"Missing {REGISTRY_FILE} in {root}")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise SourceRegistryError(f"Invalid {REGISTRY_FILE}: {exc}") from exc

    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise SourceRegistryError("Registry schema_version must be 1")
    documents = data.get("documents")
    if not isinstance(documents, dict) or not documents:
        raise SourceRegistryError("Registry documents must be a nonempty object")

    paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    source_files = {p.name: p for p in paths}
    if len(source_files) != len(paths):
        raise SourceRegistryError("Ambiguous or duplicated document filenames")
    if any(
        not isinstance(name, str)
        or Path(name).name != name
        or "\\" in name
        or "/" in name
        or name in {".", ".."}
        for name in documents
    ):
        raise SourceRegistryError("Registry keys must be simple document filenames")

    present = set(source_files)
    declared = set(documents)
    if present != declared:
        raise SourceRegistryError(
            f"Registry differs from corpus (missing={sorted(present - declared)}, "
            f"not_found={sorted(declared - present)})"
        )
    today = today or date.today()
    admitted = {}
    for filename in sorted(documents):
        entry = documents[filename]
        if not isinstance(entry, dict):
            raise SourceRegistryError(f"{filename}: entry must be an object")
        path = source_files[filename]
        if path.parent.resolve() != root.resolve() or path.is_symlink():
            raise SourceRegistryError(f"{filename}: nested/symlinked files not supported")
        title = _required(entry, "title", filename)
        doc_type = _required(entry, "document_type", filename)
        if doc_type not in DOC_TYPES:
            raise SourceRegistryError(f"{filename}: unsupported document_type")
        jurisdiction = _required(entry, "jurisdiction", filename)
        publisher = _required(entry, "publisher", filename)
        source_url = _official_url(
            _required(entry, "source_url", filename), "source_url", filename
        )
        if entry.get("legal_status") != "current":
            raise SourceRegistryError(f"{filename}: source must be reviewed as current")
        if entry.get("rights_status") != "authorized":
            raise SourceRegistryError(f"{filename}: reuse must be explicitly authorized")
        rights_basis = _required(entry, "rights_basis", filename)
        rights_reference = _required(entry, "rights_reference", filename)
        reviewer = _required(entry, "reviewer", filename)
        reviewed_on = _date(entry, "reviewed_on", filename, today)
        if date.fromisoformat(reviewed_on) < today - timedelta(days=MAX_REVIEW_AGE_DAYS):
            raise SourceRegistryError(f"{filename}: legal review is older than 90 days")

        declared_hash = _required(entry, "sha256", filename)
        if SHA256_RE.fullmatch(declared_hash) is None:
            raise SourceRegistryError(f"{filename}: sha256 must be lowercase hex")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if declared_hash != actual_hash:
            raise SourceRegistryError(f"{filename}: content SHA-256 mismatch")

        metadata = {
            "title": title,
            "document_type": doc_type,
            "jurisdiction": jurisdiction,
            "publisher": publisher,
            "source_url": source_url,
            "legal_status": "current",
            "rights_status": "authorized",
            "rights_basis": rights_basis,
            "rights_reference": rights_reference,
            "reviewer": reviewer,
            "reviewed_on": reviewed_on,
            "source_sha256": actual_hash,
            "corpus_admitted": True,
        }
        if doc_type == "statute":
            gazette_url = _official_url(
                _required(entry, "gazette_url", filename), "gazette_url", filename
            )
            checked_on = _date(entry, "gazette_checked_on", filename, today)
            if date.fromisoformat(checked_on) < today - timedelta(days=MAX_REVIEW_AGE_DAYS):
                raise SourceRegistryError(f"{filename}: Gazette check is older than 90 days")
            metadata.update({
                "effective_date": _date(entry, "effective_date", filename, today),
                "gazette_url": gazette_url,
                "gazette_checked_on": checked_on,
            })
        else:
            metadata["court"] = _required(entry, "court", filename)
            metadata["decision_date"] = _date(entry, "decision_date", filename, today)
        admitted[filename] = metadata
    return admitted


def index_metadata_admitted(metadata: dict, *, today: date | None = None) -> bool:
    """Validate stored review flags at API startup/readiness (not a legal audit)."""
    today = today or date.today()
    if (
        metadata.get("corpus_admitted") is not True
        or metadata.get("legal_status") != "current"
        or metadata.get("rights_status") != "authorized"
        or not metadata.get("source_url")
        or not metadata.get("source_sha256")
        or not metadata.get("rights_reference")
        or not metadata.get("reviewer")
    ):
        return False
    try:
        reviewed = date.fromisoformat(metadata["reviewed_on"])
        if reviewed > today or reviewed < today - timedelta(days=MAX_REVIEW_AGE_DAYS):
            return False
        if metadata.get("document_type") == "statute":
            gazette_checked = date.fromisoformat(metadata["gazette_checked_on"])
            if not metadata.get("gazette_url") or gazette_checked > today:
                return False
            if gazette_checked < today - timedelta(days=MAX_REVIEW_AGE_DAYS):
                return False
    except (KeyError, ValueError, TypeError):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a human-reviewed legal corpus registry.")
    parser.add_argument("--corpus", required=True, help="Folder containing source_registry.json and documents.")
    args = parser.parse_args(argv)
    try:
        admitted = validate_registry(args.corpus)
    except (SourceRegistryError, OSError) as exc:
        print(f"Corpus not admitted: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "admitted", "documents": sorted(admitted)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
