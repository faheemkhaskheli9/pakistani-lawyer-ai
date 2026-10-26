from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.chunk_store import ChunkRecord, ChunkStore
from legal_core.verify_tool import CitationVerifier

KEY = {"X-API-Key": "test-key"}


def make(tmp_path):
    ChunkStore(tmp_path / "chunks.json").upsert(
        [
            ChunkRecord("a", "/x/ppc.txt", "Section 302", 0, "Punishment of qatl-i-amd under the Pakistan Penal Code.", {}),
            ChunkRecord("b", "/x/notes.txt", "Note", 0, "Followed in PLD 2010 SC 265 and 2015 SCMR 100.", {}),
        ]
    )
    return CitationVerifier(tmp_path)


def test_reported_case_found_and_missing(tmp_path):
    v = make(tmp_path)
    ok = v.verify("pld  2010 sc 265")
    assert ok.kind == "reported_case" and ok.found_in_corpus and ok.normalized == "PLD 2010 SC 265"
    miss = v.verify("2001 SCMR 999")
    assert miss.format_valid and not miss.found_in_corpus


def test_statute_section(tmp_path):
    v = make(tmp_path)
    assert v.verify("Section 302 of the Pakistan Penal Code").found_in_corpus
    assert v.verify("s. 302").found_in_corpus
    assert not v.verify("Section 999 PPC").found_in_corpus
    assert not v.verify("Section 302 Contract Act").found_in_corpus


def test_unrecognised_and_empty(tmp_path):
    v = make(tmp_path)
    assert v.verify("Smith v Jones").kind == "unrecognised"
    try:
        v.verify("  ")
    except ValueError:
        pass
    else:
        raise AssertionError


def test_endpoint(tmp_path):
    c = TestClient(create_app(qa_service=object(), search_service=object(), citation_verifier=make(tmp_path), api_key="test-key"))
    body = c.get("/api/v1/verify-citation", params={"citation": "PLD 2010 SC 265"}, headers=KEY).json()
    assert body["found_in_corpus"] is True
    assert c.get("/api/v1/verify-citation", params={"citation": "PLD 2010 SC 265"}).status_code == 401
