"""Limitation-period calculator (Limitation Act, 1908 -- selected Schedule articles).

Informational only: it does not apply exclusions (sections 4-25), acknowledgments or
condonation of delay (section 5). The caller must verify against the current Act.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .safeguards import LEGAL_INFORMATION_DISCLAIMER as DISCLAIMER


@dataclass(frozen=True)
class LimitationRule:
    key: str
    article: str
    description: str
    years: int
    runs_from: str


RULES: dict[str, LimitationRule] = {
    r.key: r
    for r in (
        LimitationRule("breach_of_contract", "Article 115", "Compensation for breach of contract", 3, "when the contract is broken"),
        LimitationRule("movable_property", "Article 49", "Compensation for wrongful taking or detention of movable property", 3, "when the property is wrongfully taken or detained"),
        LimitationRule("money_had_and_received", "Article 62", "Money payable by the defendant to the plaintiff for money received", 3, "when the money is received"),
        LimitationRule("immovable_possession", "Article 144", "Possession of immovable property or any interest in it", 12, "when the defendant's possession becomes adverse"),
        LimitationRule("residuary", "Article 120", "Suit for which no period is provided elsewhere", 6, "when the right to sue accrues"),
    )
}


@dataclass(frozen=True)
class LimitationResult:
    rule: LimitationRule
    start_date: date
    expiry_date: date
    expired: bool
    days_remaining: int | None
    disclaimer: str = DISCLAIMER


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 Feb -> 28 Feb
        return d.replace(year=d.year + years, day=28)


def calculate_limitation(kind: str, start_date: date, today: date | None = None) -> LimitationResult:
    rule = RULES.get(kind)
    if rule is None:
        raise ValueError(f"Unknown suit type '{kind}'. Supported: {', '.join(sorted(RULES))}")
    today = today or date.today()
    # The day the cause accrues is excluded (s.12), so the period ends on the anniversary.
    expiry = _add_years(start_date, rule.years)
    remaining = (expiry - today).days
    return LimitationResult(
        rule=rule,
        start_date=start_date,
        expiry_date=expiry,
        expired=remaining < 0,
        days_remaining=max(remaining, 0) if remaining >= 0 else None,
    )
