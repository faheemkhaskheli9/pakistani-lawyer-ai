"""Tests for Pakistani legal citation parsing (issue #21)."""
from legal_core.citation_parser import (
    contains_supported_citation,
    find_citations,
    parse_citation,
)
from legal_core.query_analysis import classify_intent


def test_parses_pld_supreme_court_citation():
    citation = parse_citation("PLD 2024 SC 123")
    assert citation is not None
    assert citation.reporter == "PLD"
    assert citation.year == 2024
    assert citation.court == "SC"
    assert citation.page == 123
    assert citation.normalized == "PLD 2024 SC 123"


def test_parses_pld_high_court_location():
    citation = parse_citation("pld   2025   lahore   45")
    assert citation is not None
    assert citation.court == "Lahore"
    assert citation.normalized == "PLD 2025 Lahore 45"


def test_parses_supported_non_pld_reporters():
    examples = {
        "2024 SCMR 101": "SCMR",
        "2023 CLC 202": "CLC",
        "2022 MLD 303": "MLD",
        "2021 YLR 404": "YLR",
    }
    for raw, reporter in examples.items():
        citation = parse_citation(raw)
        assert citation is not None
        assert citation.reporter == reporter
        assert citation.normalized == raw


def test_invalid_or_unsupported_citation_returns_none():
    assert parse_citation("") is None
    assert parse_citation("PLD SC 123") is None
    assert parse_citation("2024 ABC 123") is None
    assert parse_citation("some legal text") is None


def test_finds_multiple_citations_in_free_text_in_order():
    found = find_citations(
        "The court considered PLD 2024 SC 123 and later 2023 SCMR 456. "
        "PLD 2024 SC 123 was cited again."
    )
    assert [item.normalized for item in found] == [
        "PLD 2024 SC 123",
        "2023 SCMR 456",
    ]


def test_supported_citation_detection_drives_query_intent():
    assert contains_supported_citation("See 2024 CLC 123 for the principle.")
    assert classify_intent("Please find 2024 CLC 123") == "citation_lookup"
