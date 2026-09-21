"""Tests for top-k retrieval over the vector store (issue #5): ranked by
similarity, configurable k, and an empty/uninitialized store yields an empty
result rather than an exception. Everything runs offline on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from legal_core.embeddings import EmbeddingProvider, HashingEmbedder
from legal_core.ingest import run_ingestion
from legal_core.retrieval import (
    DEFAULT_TOP_K,
    RetrievedChunk,
    Retriever,
    build_retriever,
    main,
    resolve_top_k,
)
from legal_core.vector_store import DimensionMismatch, VectorStore, VectorStoreCorrupt

SEED_CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


class AxisEmbedder(EmbeddingProvider):
    """Deterministic 3-d embedder: each axis counts one keyword, so the
    expected ranking of a test is obvious from the text alone."""

    KEYWORDS = ("contract", "bail", "evidence")

    def __init__(self):
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return len(self.KEYWORDS)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(text.lower().count(word)) for word in self.KEYWORDS] for text in texts]


def _store(tmp_path, embedder: AxisEmbedder, documents: dict[str, str]) -> VectorStore:
    store = VectorStore(tmp_path / "index" / "vectors")
    for chunk_id, text in documents.items():
        [vector] = embedder.embed([text])
        store.upsert(
            chunk_id, vector, text,
            metadata={"source": f"{chunk_id}.txt", "section": "Section 1", "chunk_index": 0},
        )
    embedder.calls.clear()
    return store


DOCUMENTS = {
    "contracts": "contract contract contract formation",
    "mixed": "contract and bail",
    "bail": "bail bail bail conditions",
    "evidence": "evidence admissibility",
}


# --- ranked by similarity ----------------------------------------------------------

def test_results_are_ranked_by_descending_similarity(tmp_path):
    embedder = AxisEmbedder()
    retriever = Retriever(embedder, _store(tmp_path, embedder, DOCUMENTS), top_k=4)

    results = retriever.retrieve("contract")

    assert [r.chunk_id for r in results[:2]] == ["contracts", "mixed"]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert results[0].score == pytest.approx(1.0)


def test_results_carry_text_and_source_citation(tmp_path):
    embedder = AxisEmbedder()
    [top] = Retriever(embedder, _store(tmp_path, embedder, DOCUMENTS), top_k=1).retrieve("bail")

    assert isinstance(top, RetrievedChunk)
    assert top.text == DOCUMENTS["bail"]
    assert (top.source, top.section, top.chunk_index) == ("bail.txt", "Section 1", 0)
    assert top.citation == "bail.txt, Section 1"


def test_equal_scores_rank_deterministically_in_insertion_order(tmp_path):
    embedder = AxisEmbedder()
    ties = {f"doc-{i}": "contract law" for i in range(6)}
    retriever = Retriever(embedder, _store(tmp_path, embedder, ties), top_k=6)

    first = [r.chunk_id for r in retriever.retrieve("contract")]

    assert first == list(ties)
    assert all([r.chunk_id for r in retriever.retrieve("contract")] == first for _ in range(3))


# --- k is configurable ---------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3])
def test_top_k_is_configurable_at_construction(tmp_path, k):
    embedder = AxisEmbedder()
    retriever = Retriever(embedder, _store(tmp_path, embedder, DOCUMENTS), top_k=k)
    assert len(retriever.retrieve("contract bail evidence")) == k


def test_top_k_can_be_overridden_per_call(tmp_path):
    embedder = AxisEmbedder()
    retriever = Retriever(embedder, _store(tmp_path, embedder, DOCUMENTS), top_k=1)
    assert len(retriever.retrieve("contract bail evidence", top_k=3)) == 3
    assert len(retriever.retrieve("contract bail evidence")) == 1


def test_top_k_larger_than_the_store_returns_everything(tmp_path):
    embedder = AxisEmbedder()
    retriever = Retriever(embedder, _store(tmp_path, embedder, DOCUMENTS), top_k=50)
    assert len(retriever.retrieve("contract bail evidence")) == len(DOCUMENTS)


def test_top_k_defaults_and_env_override(monkeypatch):
    assert resolve_top_k(env={}) == DEFAULT_TOP_K
    assert resolve_top_k(env={"RETRIEVAL_TOP_K": " 9 "}) == 9
    assert resolve_top_k(env={"RETRIEVAL_TOP_K": ""}) == DEFAULT_TOP_K
    assert resolve_top_k(2, env={"RETRIEVAL_TOP_K": "9"}) == 2  # explicit argument wins

    monkeypatch.setenv("RETRIEVAL_TOP_K", "7")
    assert Retriever(AxisEmbedder(), VectorStore("unused-path-never-written")).top_k == 7


@pytest.mark.parametrize("bad", ["0", "-3", "five", "2.5"])
def test_invalid_env_top_k_is_an_error_not_a_silent_default(bad):
    with pytest.raises(ValueError, match="RETRIEVAL_TOP_K"):
        resolve_top_k(env={"RETRIEVAL_TOP_K": bad})


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3"])
def test_invalid_top_k_argument_is_rejected(tmp_path, bad):
    embedder = AxisEmbedder()
    store = _store(tmp_path, embedder, DOCUMENTS)
    with pytest.raises(ValueError, match="top_k"):
        Retriever(embedder, store, top_k=bad)
    with pytest.raises(ValueError, match="top_k"):
        Retriever(embedder, store).retrieve("contract", top_k=bad)
    with pytest.raises(ValueError, match="top_k"):
        store.query([1.0, 0.0, 0.0], top_k=bad)


def test_invalid_top_k_is_rejected_even_when_the_store_is_empty(tmp_path):
    # Otherwise the mistake only surfaces after the first ingestion.
    retriever = Retriever(AxisEmbedder(), VectorStore(tmp_path / "vectors"))
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("contract", top_k=0)


# --- empty / uninitialized store -------------------------------------------------------

def test_uninitialized_index_directory_returns_empty_result(tmp_path):
    retriever = build_retriever(tmp_path / "never-created")
    assert retriever.retrieve("What is the limitation period?") == []
    assert not (tmp_path / "never-created").exists()  # and retrieval did not create it


def test_empty_store_returns_empty_result_without_embedding_the_query(tmp_path):
    embedder = AxisEmbedder()
    retriever = Retriever(embedder, VectorStore(tmp_path / "vectors"))
    assert retriever.retrieve("contract") == []
    assert embedder.calls == []


@pytest.mark.parametrize("query", ["", "   ", None])
def test_blank_query_returns_empty_result(tmp_path, query):
    embedder = AxisEmbedder()
    retriever = Retriever(embedder, _store(tmp_path, embedder, DOCUMENTS))
    assert retriever.retrieve(query) == []
    assert embedder.calls == []


def test_query_with_no_signal_returns_empty_result(tmp_path):
    embedder = AxisEmbedder()  # "tenancy" hits none of its axes -> zero vector
    retriever = Retriever(embedder, _store(tmp_path, embedder, DOCUMENTS))
    assert retriever.retrieve("tenancy") == []


# --- loud failures for real mistakes -----------------------------------------------------

def test_querying_with_a_different_embedding_provider_is_a_clear_error(tmp_path):
    embedder = AxisEmbedder()
    store = _store(tmp_path, embedder, DOCUMENTS)  # 3-d index
    with pytest.raises(DimensionMismatch, match="different embedding provider"):
        Retriever(HashingEmbedder(dimension=16), store).retrieve("contract")


def test_store_whose_vectors_and_metadata_disagree_refuses_to_load(tmp_path):
    embedder = AxisEmbedder()
    store = _store(tmp_path, embedder, DOCUMENTS)
    # Simulate a crash between the two file replacements: vectors are one
    # chunk ahead of the metadata.
    with store._vectors_path().open("wb") as fh:
        np.savez(fh, vectors=np.vstack([store._vectors, [[0.0, 0.0, 1.0]]]))

    with pytest.raises(VectorStoreCorrupt, match="inconsistent"):
        VectorStore(store.path)


def test_store_with_mismatched_ids_and_documents_refuses_to_load(tmp_path):
    embedder = AxisEmbedder()
    store = _store(tmp_path, embedder, DOCUMENTS)
    meta = json.loads(store._meta_path().read_text(encoding="utf-8"))
    meta["ids"][0] = "some-other-chunk"
    store._meta_path().write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(VectorStoreCorrupt):
        VectorStore(store.path)


# --- end to end against the seed corpus + CLI ---------------------------------------------

@pytest.fixture()
def seed_index(tmp_path):
    index_dir = tmp_path / "index"
    ingested, _ = run_ingestion(SEED_CORPUS, index_dir, embedding_provider="hashing")
    assert ingested > 0
    return index_dir


def test_retrieval_over_the_ingested_seed_corpus(seed_index):
    results = build_retriever(seed_index, top_k=3, embedding_provider="hashing").retrieve(
        "admissibility of evidence"
    )
    assert len(results) == 3
    assert all(r.source and r.section and r.text for r in results)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_cli_prints_ranked_results_with_citations(seed_index, capsys):
    code = main(["admissibility of evidence", "--index-dir", str(seed_index), "--top-k", "2"])
    out = capsys.readouterr().out
    assert code == 0
    lines = [line for line in out.splitlines() if line[:2] in {"1.", "2.", "3."}]
    assert [line[:2] for line in lines] == ["1.", "2."]


def test_cli_on_an_unbuilt_index_explains_instead_of_failing(tmp_path, capsys):
    code = main(["anything", "--index-dir", str(tmp_path / "nope")])
    assert code == 0
    assert "index" in capsys.readouterr().out.lower()


def test_cli_rejects_a_bad_top_k_with_a_nonzero_exit(seed_index, capsys):
    assert main(["evidence", "--index-dir", str(seed_index), "--top-k", "0"]) == 1
    assert "top_k" in capsys.readouterr().err
