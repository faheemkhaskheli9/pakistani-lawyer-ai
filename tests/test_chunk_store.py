from __future__ import annotations

from legal_core.chunk_store import ChunkRecord, ChunkStore


def make_record(rid="id-1", text="hello"):
    return ChunkRecord(id=rid, source="doc.txt", section="Section 1", chunk_index=0, text=text)


def test_upsert_then_reload_persists_records(tmp_path):
    path = tmp_path / "chunks.json"
    store = ChunkStore(path)
    store.upsert([make_record()])

    reloaded = ChunkStore(path)
    assert len(reloaded) == 1
    assert reloaded.get_ids() == {"id-1"}


def test_upsert_same_id_replaces_not_appends(tmp_path):
    store = ChunkStore(tmp_path / "chunks.json")
    store.upsert([make_record(text="version 1")])
    store.upsert([make_record(text="version 2")])

    assert len(store) == 1
    assert store.all()[0].text == "version 2"


def test_missing_file_starts_empty(tmp_path):
    store = ChunkStore(tmp_path / "does-not-exist-yet.json")
    assert len(store) == 0
