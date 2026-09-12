from __future__ import annotations

import pytest

from legal_core.parsing import DocumentParseError, read_source_text


def test_reads_txt_file(tmp_path):
    path = tmp_path / "statute.txt"
    path.write_text("Section 1. Short title.", encoding="utf-8")
    assert read_source_text(path) == "Section 1. Short title."


def test_missing_file_raises_document_parse_error(tmp_path):
    with pytest.raises(DocumentParseError):
        read_source_text(tmp_path / "does-not-exist.txt")


def test_unsupported_extension_raises_document_parse_error(tmp_path):
    path = tmp_path / "statute.docx"
    path.write_text("not supported", encoding="utf-8")
    with pytest.raises(DocumentParseError):
        read_source_text(path)


def test_undecodable_bytes_do_not_crash_the_caller(tmp_path):
    # .txt is read with errors="replace" so this must not raise at all.
    path = tmp_path / "bad-encoding.txt"
    path.write_bytes(b"\xff\xfe\x00Section 1")
    text = read_source_text(path)
    assert isinstance(text, str)


def test_corrupt_pdf_raises_document_parse_error_not_a_crash(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4 this is not a real pdf stream")
    with pytest.raises(DocumentParseError):
        read_source_text(path)
