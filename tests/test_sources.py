import json

import pytest

from legal_core.sources import HttpSourceAdapter, SourceError, acquire, html_to_text, validate_entry

ENTRY = {
    "name": "act_one", "title": "Act One", "url": "https://example.gov.pk/a.html", "format": "html",
    "license": "public-domain", "license_evidence": "https://example.gov.pk/terms",
    "jurisdiction": "Federal", "document_type": "statute",
}


def test_html_to_text_strips_markup_and_scripts():
    text = html_to_text("<html><head><title>x</title></head><body><script>bad()</script>"
                        "<h1>Act</h1><p>Section 1. Hello   world</p></body></html>")
    assert text == "Act\nSection 1. Hello world"


@pytest.mark.parametrize("patch", [{"license": "all-rights-reserved"}, {"license": ""},
                                   {"url": "http://insecure.example/a"}, {"name": "../evil"},
                                   {"format": "docx"}, {"license_evidence": ""}])
def test_validate_entry_rejects_bad_entries(patch):
    with pytest.raises(SourceError):
        validate_entry({**ENTRY, **patch})


def test_acquire_writes_text_and_provenance(tmp_path):
    adapter = HttpSourceAdapter(lambda url: b"<p>Section 1. Test law.</p>")
    names = acquire([ENTRY], tmp_path, adapter=adapter, today="2026-01-01")
    assert names == ["act_one"]
    assert (tmp_path / "act_one.txt").read_text() == "Section 1. Test law.\n"
    record = json.loads((tmp_path / "sources.json").read_text())["act_one.txt"]
    assert record["license"] == "public-domain" and record["retrieved"] == "2026-01-01"
    assert record["jurisdiction"] == "Federal" and len(record["sha256"]) == 64


def test_license_gate_runs_before_any_download(tmp_path):
    calls = []
    adapter = HttpSourceAdapter(lambda url: calls.append(url) or b"x")
    with pytest.raises(SourceError):
        acquire([ENTRY, {**ENTRY, "name": "b", "license": "proprietary"}], tmp_path, adapter=adapter)
    assert calls == []


def test_acquire_preserves_existing_sources(tmp_path):
    (tmp_path / "sources.json").write_text(json.dumps({"keep.txt": {"title": "Keep"}}))
    acquire([ENTRY], tmp_path, adapter=HttpSourceAdapter(lambda u: b"<p>text</p>"))
    assert "keep.txt" in json.loads((tmp_path / "sources.json").read_text())


def test_acquired_provenance_feeds_ingestion_metadata(tmp_path):
    from legal_core.metadata import load_legal_metadata

    acquire([ENTRY], tmp_path, adapter=HttpSourceAdapter(lambda u: b"<p>Section 1. Law.</p>"))
    meta = load_legal_metadata(tmp_path / "act_one.txt", text="Section 1. Law.")
    assert meta.jurisdiction == "Federal" and meta.document_type == "statute"
    assert meta.source_url == ENTRY["url"]
