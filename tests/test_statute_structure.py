"""Tests for statute hierarchy parsing and ingestion (issue #24)."""
from legal_core.chunk_store import ChunkStore
from legal_core.ingest import IngestionPipeline
from legal_core.statute_structure import (
    StatuteStructureTracker,
    extract_statute_structure,
)


def test_extracts_common_statute_hierarchy_labels():
    structure = extract_statute_structure(
        """PART II
CHAPTER IV
5. Free consent.
(1) Consent is free when...
(a) coercion is absent.
Explanation.—This explanation applies.
"""
    )
    assert structure.part == "Part II"
    assert structure.chapter == "Chapter IV"
    assert structure.section == "Section 5"
    assert structure.subsection == "Subsection (1)"
    assert structure.clause == "Clause (a)"
    assert structure.explanation == "Explanation"


def test_detects_schedule_heading():
    structure = extract_statute_structure("FIRST SCHEDULE\nPrescribed forms")
    assert structure.schedule == "First Schedule"


def test_tracker_carries_hierarchy_across_sequential_chunks():
    tracker = StatuteStructureTracker()
    first = tracker.update("CHAPTER III\n12. Appeals.")
    second = tracker.update("Further text continuing the provision.")
    assert first.chapter == "Chapter III"
    assert first.section == "Section 12"
    assert second.chapter == "Chapter III"
    assert second.section == "Section 12"


def test_new_section_resets_subsection_and_clause():
    tracker = StatuteStructureTracker()
    tracker.update("5. First section.\n(1) Text\n(a) Clause")
    second = tracker.update("6. Second section.")
    assert second.section == "Section 6"
    assert second.subsection is None
    assert second.clause is None


def test_unstructured_text_has_empty_structure():
    assert extract_statute_structure("ordinary prose without headings").to_dict() == {}


def test_ingestion_preserves_structure_metadata(tmp_path):
    source = tmp_path / "sample_act.txt"
    source.write_text(
        "CHAPTER I\n1. Short title.\nThis Act may be called the Demo Act.",
        encoding="utf-8",
    )
    store = ChunkStore(tmp_path / "chunks.json")
    IngestionPipeline(store, chunk_size=500, chunk_overlap=0).ingest_file(source)

    record = store.all()[0]
    structure = record.metadata["statute_structure"]
    assert structure["chapter"] == "Chapter I"
    assert structure["section"] == "Section 1"
