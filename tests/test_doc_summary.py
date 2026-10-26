import pytest
from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.doc_summary import guess_kind, summarize_document

FIR = (
    "First Information Report. FIR No. 123/2024 registered at Police Station Saddar. "
    "The complainant states that on 12/03/2024 the accused snatched Rs. 50,000 at gunpoint. "
    "The case is registered under Section 392 PPC and sections 337A, 34 PPC. "
    "Reference: PLD 2010 SC 265. Short. The complainant requests legal action against the accused persons."
)


def test_extracts_entities_and_kind():
    s = summarize_document(FIR)
    assert s.kind == "fir"
    assert "12/03/2024" in s.dates and any("50,000" in a for a in s.amounts)
    assert "Section 392 PPC" in s.sections
    assert s.case_citations and "PLD 2010" in s.case_citations[0]


def test_points_are_verbatim_with_offsets():
    s = summarize_document(FIR, max_points=3)
    assert 1 <= len(s.key_points) <= 3
    for p in s.key_points:
        assert FIR[p.start:p.end] == p.text
    starts = [p.start for p in s.key_points]
    assert starts == sorted(starts)


def test_empty_and_too_large():
    with pytest.raises(ValueError):
        summarize_document("   ")
    with pytest.raises(ValueError):
        summarize_document("x" * 200_001)
    assert guess_kind("hello") == "unknown"


def test_endpoint():
    c = TestClient(create_app(qa_service=object(), search_service=object(), api_key="k"))
    assert c.post("/api/v1/documents/summary", json={"text": FIR}).status_code == 401
    r = c.post("/api/v1/documents/summary", json={"text": FIR}, headers={"X-API-Key": "k"})
    assert r.status_code == 200 and r.json()["kind"] == "fir" and r.json()["disclaimer"]
    assert c.post("/api/v1/documents/summary", json={"text": "  "}, headers={"X-API-Key": "k"}).status_code == 400
