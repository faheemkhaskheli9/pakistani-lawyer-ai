"""Source-document text extraction for corpus ingestion.

Supports `.txt` and `.pdf`. The `pypdf` package is imported lazily inside
`_read_pdf` so it's only a hard dependency when a PDF is actually parsed.
"""
from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


class DocumentParseError(Exception):
    """Raised when a source document can't be parsed into usable text.

    Callers (see `legal_core.ingest`) must catch this per-file and skip the
    offending document with a logged error rather than let it crash the
    whole ingestion run.
    """


def read_source_text(path: str | Path) -> str:
    """Extract plain text from a `.txt` or `.pdf` source document.

    Raises `DocumentParseError` for an unsupported extension, a missing
    file, or a file that fails to parse (corrupt/encrypted PDF, undecodable
    bytes) — never lets the underlying exception type leak to callers.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(f"Unsupported source extension: {path.suffix!r} ({path})")
    if not path.exists():
        raise DocumentParseError(f"Source file not found: {path}")

    try:
        if suffix == ".txt":
            return _read_txt(path)
        return _read_pdf(path)
    except DocumentParseError:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any parse
        # failure (bad encoding, corrupt PDF structure, ...) must become a
        # DocumentParseError so ingestion can skip-and-log instead of crash.
        raise DocumentParseError(f"Failed to parse {path}: {exc}") from exc


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise DocumentParseError(
            "PDF parsing requires the 'pypdf' package: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise DocumentParseError(f"Encrypted PDF not supported: {path}")
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise DocumentParseError(f"No extractable text in PDF: {path}")
    return text
