from __future__ import annotations

import pytest

from legal_core.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_single_chunk():
    assert chunk_text("hello world", chunk_size=800) == ["hello world"]


def test_long_text_splits_into_multiple_chunks():
    text = "word " * 500  # far longer than chunk_size
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_boundaries_do_not_split_words():
    # chunk_size comfortably larger than any single word so the whitespace
    # snap always finds a boundary — a word longer than chunk_size is an
    # inherent, separate edge case (fixed-size chunking with no boundary to
    # snap to), not what this test is checking.
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
    for chunk in chunks:
        assert not chunk.startswith(" ")
    rejoined_words = " ".join(chunks).split()
    original_words = text.split()
    assert set(rejoined_words) <= set(original_words)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=10, chunk_overlap=10)


def test_overlap_near_chunk_size_does_not_infinite_loop():
    # Regression: overlap close to chunk_size combined with a whitespace
    # snap-back must still make forward progress each iteration.
    text = "a " * 1000
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=49)
    assert len(chunks) > 0  # completes at all = no infinite loop
