"""Structured legal-document metadata used during ingestion and retrieval."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

LEGAL_METADATA_MANIFEST = "legal_metadata.json"
SOURCES_MANIFEST = "sources.json"


@dataclass(frozen=True)
class LegalMetadata:
    title: str | None = None
    document_type: str | None = None
    jurisdiction: str | None = None
    court: str | None = None
    case_citation: str | None = None
    decision_date: str | None = None
    effective_date: str | None = None
    source_url: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata manifest must be a JSON object: {path}")
    return data


def _validate_iso_date(value: str | None, field_name: str, source: Path) -> str | None:
    if value is None or value == "":
        return None
    try:
        date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(
            f"{field_name} for {source.name!r} must use YYYY-MM-DD ISO format"
        ) from None
    return str(value)


def infer_document_type(path: Path, text: str = "") -> str | None:
    haystack = f"{path.stem} {text[:300]}".lower()
    if any(term in haystack for term in ("judgment", "judgement", "case report")):
        return "judgment"
    if any(term in haystack for term in ("act", "code", "ordinance", "constitution", "rules")):
        return "statute"
    if "regulation" in haystack:
        return "regulation"
    return None


def load_legal_metadata(path: str | Path, *, text: str = "") -> LegalMetadata:
    """Load optional sidecar metadata and merge it with the source manifest.

    legal_metadata.json and sources.json are both optional for general
    corpora. The checked-in seed corpus already has sources.json, so it gains
    title/source provenance without requiring a new manifest.
    """
    path = Path(path)
    directory = path.parent
    source_manifest = _read_manifest(directory / SOURCES_MANIFEST)
    legal_manifest = _read_manifest(directory / LEGAL_METADATA_MANIFEST)

    source_entry = source_manifest.get(path.name, {})
    legal_entry = legal_manifest.get(path.name, {})
    if source_entry and not isinstance(source_entry, dict):
        raise ValueError(f"Invalid source metadata entry for {path.name!r}")
    if legal_entry and not isinstance(legal_entry, dict):
        raise ValueError(f"Invalid legal metadata entry for {path.name!r}")

    known = {
        "title",
        "document_type",
        "jurisdiction",
        "court",
        "case_citation",
        "decision_date",
        "effective_date",
        "source_url",
    }
    unknown = set(legal_entry) - known
    if unknown:
        raise ValueError(
            f"Unknown legal metadata field(s) for {path.name!r}: {sorted(unknown)}"
        )

    merged = {
        "title": source_entry.get("title") or path.stem,
        "source_url": source_entry.get("source_url"),
        "document_type": infer_document_type(path, text),
    }
    merged.update({key: value for key, value in legal_entry.items() if value is not None})
    merged["decision_date"] = _validate_iso_date(
        merged.get("decision_date"), "decision_date", path
    )
    merged["effective_date"] = _validate_iso_date(
        merged.get("effective_date"), "effective_date", path
    )
    return LegalMetadata(**merged)
