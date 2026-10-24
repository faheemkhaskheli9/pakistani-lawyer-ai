"""Language detection and Urdu / Roman-Urdu query normalisation (issue #31).

Deterministic and offline. Urdu (Arabic script) queries are normalised so that
visually identical spellings compare equal; Roman-Urdu queries are expanded
with the English legal terms the corpus is indexed under, because the default
embedder is lexical and cannot bridge scripts or transliterations by itself.
"""
from __future__ import annotations

import re
import unicodedata

LANG_ENGLISH = "english"
LANG_URDU = "urdu"
LANG_ROMAN_URDU = "roman_urdu"

_ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN_WORD = re.compile(r"[a-z]+")

# Marks removed during normalisation: harakat, superscript alef, tatweel.
_STRIP = dict.fromkeys(
    list(range(0x064B, 0x0653)) + [0x0670, 0x0640, 0x200C, 0x200D, 0x200F, 0x200E], None
)
_UNIFY = {
    "ي": "ی",  # Arabic yeh -> Farsi/Urdu yeh
    "ى": "ی",
    "ك": "ک",  # Arabic kaf -> Urdu kaf
    "ة": "ہ",
    "ھ": "ہ",  # heh doachashmee used as plain heh in many sources
}
_DIGITS = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGITS.update({ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})

# Words that signal Roman Urdu (rare in English, common in transliterated Urdu).
_ROMAN_URDU_MARKERS = frozenset(
    "kya hai hain ka ki ke ko se mein me nahi nahin kaise kab kyun kyu aur ya "
    "agar toh to koi kuch mujhe mera meri hum aap kar karna sakta sakti hota "
    "hoti hoga liye wala wali par pe".split()
)

# Roman-Urdu legal vocabulary -> English terms used in the corpus.
_ROMAN_URDU_LEGAL = {
    "dafa": "section",
    "dafaa": "section",
    "zamanat": "bail",
    "zamant": "bail",
    "talaq": "divorce",
    "khula": "khula dissolution of marriage",
    "nikah": "marriage",
    "nikahnama": "marriage registration",
    "wirasat": "inheritance succession",
    "virasat": "inheritance succession",
    "jaidad": "property",
    "qabza": "possession",
    "muqadma": "case suit",
    "muqaddama": "case suit",
    "mukadma": "case suit",
    "adalat": "court",
    "ahmiyat": "importance",
    "jurm": "offence crime",
    "saza": "punishment sentence",
    "qatl": "murder",
    "chori": "theft",
    "dhoka": "cheating fraud",
    "dhokadehi": "cheating fraud",
    "ikrarnama": "agreement contract",
    "muahida": "contract",
    "maahida": "contract",
    "gawahi": "witness evidence",
    "gawah": "witness",
    "saboot": "evidence",
    "shahadat": "evidence",
    "fir": "FIR first information report",
    "hukm": "order",
    "apeel": "appeal",
    "daawa": "suit claim",
    "dawa": "suit claim",
    "wakeel": "advocate lawyer",
    "ayeen": "constitution",
    "aain": "constitution",
    "qanoon": "law act",
    "qanun": "law act",
    "haq": "right",
    "huqooq": "rights",
}


def normalize_urdu(text: str) -> str:
    """Canonicalise Urdu text: NFKC, unify look-alike letters, strip marks, ASCII digits."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_STRIP)
    text = "".join(_UNIFY.get(ch, ch) for ch in text)
    return text.translate(_DIGITS)


def detect_language(text: str) -> str:
    """Return ``urdu``, ``roman_urdu`` or ``english`` for a query."""
    letters = [c for c in text if c.isalpha()]
    if letters:
        arabic = sum(1 for c in letters if _ARABIC_SCRIPT.match(c))
        if arabic / len(letters) >= 0.25:
            return LANG_URDU
    words = _LATIN_WORD.findall(text.lower())
    if not words:
        return LANG_ENGLISH
    hits = sum(1 for w in words if w in _ROMAN_URDU_MARKERS)
    legal = sum(1 for w in words if w in _ROMAN_URDU_LEGAL and w != "fir")
    if legal and (hits or len(words) <= 2):
        return LANG_ROMAN_URDU
    if hits >= 2 or (hits >= 1 and len(words) <= 3):
        return LANG_ROMAN_URDU
    return LANG_ENGLISH


def expand_roman_urdu(text: str) -> str:
    """Append English legal terms for any Roman-Urdu legal vocabulary found."""
    seen: list[str] = []
    for word in _LATIN_WORD.findall(text.lower()):
        english = _ROMAN_URDU_LEGAL.get(word)
        if english and english not in seen:
            seen.append(english)
    return f"{text} {' '.join(seen)}" if seen else text
