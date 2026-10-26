"""Concurrent writers must merge, not clobber each other (issue #30)."""
import multiprocessing as mp

from legal_core.chunk_store import ChunkRecord, ChunkStore
from legal_core.vector_store import VectorStore

N = 15


def _chunk_worker(path, worker):
    store = ChunkStore(path)  # stale snapshot taken before others write
    for i in range(N):
        store.upsert([ChunkRecord(f"w{worker}-{i}", "s.txt", "S", i, "t")])


def _vector_worker(path, worker):
    store = VectorStore(path)
    for i in range(N):
        store.upsert(f"w{worker}-{i}", [float(worker), float(i), 1.0], f"doc {worker} {i}")


def _run(target, path, workers=3):
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=target, args=(path, w)) for w in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0


def test_chunk_store_concurrent_upserts_keep_all_records(tmp_path):
    path = tmp_path / "chunks.json"
    _run(_chunk_worker, path)
    assert len(ChunkStore(path)) == 3 * N


def test_vector_store_concurrent_upserts_keep_all_rows(tmp_path):
    path = tmp_path / "vectors"
    _run(_vector_worker, path)
    store = VectorStore(path)
    assert len(store) == 3 * N
    assert len(store.entries()) == 3 * N


def test_stale_instance_does_not_drop_other_writers_records(tmp_path):
    path = tmp_path / "chunks.json"
    stale = ChunkStore(path)
    ChunkStore(path).upsert([ChunkRecord("a", "s", "S", 0, "x")])
    stale.upsert([ChunkRecord("b", "s", "S", 1, "y")])
    assert ChunkStore(path).get_ids() == {"a", "b"}
