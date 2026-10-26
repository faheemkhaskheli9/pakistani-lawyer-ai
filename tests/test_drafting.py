import pytest
from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.drafting import TEMPLATES, render_draft

KEY = {"X-API-Key": "test-key"}


def client():
    return TestClient(create_app(qa_service=object(), search_service=object(), api_key="test-key"))


def test_missing_fields_are_marked_not_invented():
    d = render_draft("affidavit", {"deponent_name": "Ali Khan"})
    assert "Ali Khan" in d.text
    assert "[[CNIC]]" in d.text and "cnic" in d.missing_fields
    assert "deponent_name" not in d.missing_fields


def test_full_fill_has_no_placeholders():
    vals = {f: f"v-{f}" for f in TEMPLATES["bail_application"].fields}
    d = render_draft("bail_application", vals)
    assert d.missing_fields == () and "[[" not in d.text
    assert "section 497" in d.text.lower()


def test_unknown_fields_ignored_and_braces_safe():
    d = render_draft("affidavit", {"statement": "{x} {0}", "evil": "{__class__}"})
    assert "{x} {0}" in d.text and "evil" not in d.text


def test_unknown_template():
    with pytest.raises(ValueError):
        render_draft("nope")


def test_every_template_renders_blank():
    for key in TEMPLATES:
        assert render_draft(key).missing_fields == TEMPLATES[key].fields


def test_api():
    c = client()
    assert c.get("/api/v1/drafts/templates").status_code == 401
    listing = c.get("/api/v1/drafts/templates", headers=KEY).json()["templates"]
    assert {t["key"] for t in listing} == set(TEMPLATES)
    r = c.post("/api/v1/drafts", json={"template": "legal_notice_recovery", "values": {"amount": "Rs. 500,000"}}, headers=KEY)
    assert r.status_code == 200 and "Rs. 500,000" in r.json()["text"] and r.json()["disclaimer"]
    assert c.post("/api/v1/drafts", json={"template": "zzz"}, headers=KEY).status_code == 400


def test_court_templates_present_and_fill():
    for key, needle in (("plaint", "Order VII"), ("written_statement", "Order VIII"), ("writ_petition", "ARTICLE 199")):
        t = TEMPLATES[key]
        d = render_draft(key, {f: f"<{f}>" for f in t.fields})
        assert needle in d.text and d.missing_fields == () and "[[" not in d.text
