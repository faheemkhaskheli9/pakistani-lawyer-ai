from __future__ import annotations

from legal_core.citation import extract_section_id


def test_extracts_section_marker():
    assert extract_section_id("Section 5 of the Act provides that...", 0) == "Section 5"


def test_extracts_article_marker_case_insensitive():
    assert extract_section_id("under article 199 of the constitution", 3) == "Article 199"


def test_extracts_alphanumeric_section_number():
    assert extract_section_id("See Section 10A for the exception.", 0) == "Section 10A"


def test_falls_back_to_positional_id_when_no_marker():
    assert extract_section_id("plain prose with no structure markers", 4) == "para-5"
