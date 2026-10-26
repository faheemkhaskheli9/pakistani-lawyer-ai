from legal_core.citation_verification import CitationCheck, CitationVerificationReport
from legal_core.export import export_markdown
from legal_core.qa import AnswerCitation, LegalAnswer
from legal_core.service import QAResult
from tests.test_api import TEST_API_KEY, client


def _result(verified=True):
    cite = AnswerCitation("c1", "act.txt", "Section 5", "act.txt, Section 5", 0.9, "Line one.\n\nLine two.")
    return QAResult(
        status="answered", message="ok",
        answer=LegalAnswer(question="Q?", answer="A.", citations=(cite,)),
        citation_verification=CitationVerificationReport(
            verified=verified, checks=(CitationCheck("c1", verified, "r"),),
            total_citations=1, verified_citations=int(verified),
        ),
    )


def test_markdown_has_answer_citations_and_disclaimer():
    md = export_markdown(_result())
    assert "**Question:** Q?" in md and "act.txt, Section 5" in md
    assert "verified against source" in md and "> Line one." in md and "\n>\n" in md
    assert md.rstrip().endswith("_" + _result().disclaimer + "_")


def test_unverified_flagged_and_no_answer_case():
    assert "NOT verified" in export_markdown(_result(verified=False))
    md = export_markdown(QAResult(status="no_relevant_source", message="Nothing found.", answer=None))
    assert "Nothing found." in md and "## Citations" not in md


def test_export_endpoint():
    c = client()
    assert c.post("/api/v1/ask/export", json={"question": "x"}).status_code == 401
    r = c.post("/api/v1/ask/export", json={"question": "What?"}, headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    assert "attachment" in r.headers["content-disposition"] and "sample.txt, Section 1" in r.text
