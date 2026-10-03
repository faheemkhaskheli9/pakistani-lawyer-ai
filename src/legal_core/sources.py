"""Public-source acquisition adapters with mandatory license tracking.

A *manifest* lists documents to acquire. Each entry must declare a license
from an explicit allowlist; anything else is rejected before any download so
copyrighted text can never enter the corpus by accident. Acquired documents
are written as `.txt` files plus a provenance record in `sources.json`
(url, license, retrieval date, content hash) that the ingestion step reads.

Manifest entry::

    {"name": "constitution_extract", "title": "...", "url": "https://...",
     "format": "html|text|pdf", "license": "public-domain",
     "license_evidence": "https://... page stating the licence",
     "jurisdiction": "Federal", "document_type": "statute"}
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
from abc import ABC, abstractmethod
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

ALLOWED_LICENSES = frozenset(
    {"public-domain", "cc0-1.0", "cc-by-4.0", "government-open-data", "original-sample"}
)
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_BYTES = 20 * 1024 * 1024


class SourceError(ValueError):
    """Raised for an invalid manifest entry or a failed/blocked acquisition."""


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()


def validate_entry(entry: dict) -> dict:
    """Return a normalised manifest entry or raise `SourceError`."""
    for key in ("name", "title", "url", "license", "license_evidence"):
        if not str(entry.get(key) or "").strip():
            raise SourceError(f"manifest entry is missing required field {key!r}")
    name = str(entry["name"]).strip()
    if not _NAME_RE.match(name):
        raise SourceError(f"invalid name {name!r}: use letters, digits, '-' and '_' only")
    license_id = str(entry["license"]).strip().lower()
    if license_id not in ALLOWED_LICENSES:
        raise SourceError(
            f"license {entry['license']!r} for {name!r} is not allowed "
            f"(allowed: {', '.join(sorted(ALLOWED_LICENSES))})"
        )
    url = str(entry["url"]).strip()
    if not url.startswith("https://"):
        raise SourceError(f"{name!r}: only https:// URLs are supported")
    fmt = str(entry.get("format", "text")).strip().lower()
    if fmt not in {"text", "html", "pdf"}:
        raise SourceError(f"{name!r}: unsupported format {fmt!r}")
    return {**entry, "name": name, "license": license_id, "url": url, "format": fmt}


Fetcher = Callable[[str], bytes]


def default_fetcher(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "pakistani-lawyer-ai/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - https enforced
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise SourceError(f"{url} exceeds the {MAX_BYTES} byte limit")
    return data


class SourceAdapter(ABC):
    @abstractmethod
    def fetch_text(self, entry: dict) -> str:
        """Return plain text for a validated manifest entry."""


class HttpSourceAdapter(SourceAdapter):
    def __init__(self, fetcher: Fetcher = default_fetcher):
        self.fetcher = fetcher

    def fetch_text(self, entry: dict) -> str:
        raw = self.fetcher(entry["url"])
        if entry["format"] == "pdf":
            text = pdf_to_text(raw)
        else:
            decoded = raw.decode("utf-8", errors="replace")
            text = html_to_text(decoded) if entry["format"] == "html" else decoded
        if not text.strip():
            raise SourceError(f"{entry['name']!r}: no text could be extracted from {entry['url']}")
        return text.strip() + "\n"


def acquire(
    manifest: list[dict],
    out_dir: str | Path,
    *,
    adapter: SourceAdapter | None = None,
    today: str | None = None,
) -> list[str]:
    """Download manifest documents into `out_dir` and record provenance.

    All entries are validated (license gate) before any network access.
    Returns the names written.
    """
    entries = [validate_entry(entry) for entry in manifest]
    adapter = adapter or HttpSourceAdapter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources_path = out / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}

    written = []
    for entry in entries:
        text = adapter.fetch_text(entry)
        filename = f"{entry['name']}.txt"
        (out / filename).write_text(text, encoding="utf-8")
        sources[filename] = {
            "title": entry["title"],
            "source_url": entry["url"],
            "license": entry["license"],
            "license_evidence": entry["license_evidence"],
            "retrieved": today or date.today().isoformat(),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            **{k: entry[k] for k in ("jurisdiction", "document_type", "court", "date") if entry.get(k)},
        }
        written.append(entry["name"])
    sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire licensed public legal sources into the corpus.")
    parser.add_argument("manifest", help="JSON manifest listing documents to acquire")
    parser.add_argument("--out", default="data/corpus")
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    names = acquire(manifest, args.out)
    print(f"Acquired {len(names)} document(s): {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
