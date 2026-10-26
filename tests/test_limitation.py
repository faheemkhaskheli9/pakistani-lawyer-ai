from datetime import date

import pytest
from fastapi.testclient import TestClient

from legal_core.api import create_app
from legal_core.limitation import calculate_limitation

KEY = {"X-API-Key": "test-key"}


def test_contract_three_years():
    r = calculate_limitation("breach_of_contract", date(2022, 3, 10), today=date(2024, 1, 1))
    assert r.expiry_date == date(2025, 3, 10)
    assert not r.expired
    assert r.days_remaining == (date(2025, 3, 10) - date(2024, 1, 1)).days


def test_expired():
    r = calculate_limitation("residuary", date(2010, 1, 1), today=date(2020, 1, 1))
    assert r.expired and r.days_remaining is None


def test_leap_day_start():
    r = calculate_limitation("breach_of_contract", date(2020, 2, 29), today=date(2021, 1, 1))
    assert r.expiry_date == date(2023, 2, 28)


def test_unknown_kind():
    with pytest.raises(ValueError):
        calculate_limitation("nope", date(2020, 1, 1))


def client():
    return TestClient(create_app(qa_service=object(), search_service=object(), api_key="test-key"))


def test_endpoint():
    body = client().get("/api/v1/limitation", params={"kind": "immovable_possession", "start": "2000-01-01"}, headers=KEY).json()
    assert body["expiry_date"] == "2012-01-01" and body["rule"]["article"] == "Article 144"
    assert "disclaimer" in body


def test_endpoint_errors():
    c = client()
    assert c.get("/api/v1/limitation", params={"kind": "x", "start": "2000-01-01"}, headers=KEY).status_code == 400
    assert c.get("/api/v1/limitation", params={"kind": "residuary", "start": "2000-01-01"}).status_code == 401
    assert len(c.get("/api/v1/limitation/rules", headers=KEY).json()["rules"]) == 5
