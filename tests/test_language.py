import pytest

from legal_core.language import (
    LANG_ENGLISH,
    LANG_ROMAN_URDU,
    LANG_URDU,
    detect_language,
    expand_roman_urdu,
    normalize_urdu,
)
from legal_core.query_analysis import analyze_query


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What is bail under CrPC section 497?", LANG_ENGLISH),
        ("FIR kya hai", LANG_ROMAN_URDU),
        ("zamanat kaise milti hai", LANG_ROMAN_URDU),
        ("talaq ka tareeqa", LANG_ROMAN_URDU),
        ("ضمانت کیا ہے؟", LANG_URDU),
        ("Section 302 PPC ضمانت", LANG_URDU),
        ("Define contract", LANG_ENGLISH),
        ("", LANG_ENGLISH),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_normalize_urdu_unifies_letters_digits_and_marks():
    assert normalize_urdu("كيا") == "کیا"
    assert normalize_urdu("دفعہ ۳۰۲") == "دفعہ 302"
    assert normalize_urdu("ضَمانت") == "ضمانت"
    assert normalize_urdu("قــانون") == "قانون"


def test_expand_roman_urdu_adds_english_terms_once():
    out = expand_roman_urdu("zamanat ki dafa")
    assert out.startswith("zamanat ki dafa")
    assert "bail" in out and "section" in out
    assert expand_roman_urdu("hello world") == "hello world"


def test_analyze_query_records_language_and_expands():
    a = analyze_query("zamanat kaise milti hai")
    assert a.language == LANG_ROMAN_URDU and "bail" in a.normalized_query
    b = analyze_query("دفعہ ۳۰۲")
    assert b.language == LANG_URDU and "302" in b.normalized_query
    assert analyze_query("What is bail?").language == LANG_ENGLISH
