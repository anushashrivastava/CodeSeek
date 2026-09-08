"""Unit tests for RRFRanker: BM25 + cosine fusion via Reciprocal Rank
Fusion (rank-based, not score-based).

Uses the same offline-testing pattern as test_hybrid_ranker.py: a fake
embedder returning hand-built vectors, so these tests run without a
real model. What's under test here is the *rank-fusion arithmetic*
(the RRF formula, missing-rank handling, sorting), not real-world
ranking quality -- that's evaluate.py's job.
"""

import numpy as np
import pytest

from backend.embeddings.vector_store import VectorStore
from backend.indexing.inverted_index import InvertedIndex
from backend.parser.models import CodeUnit
from backend.retrieval.rrf_ranker import DEFAULT_RRF_K, RRFRanker


def _make_unit(unit_id: int, name: str, source: str) -> CodeUnit:
    return CodeUnit(
        id=unit_id,
        file_path=f"{name}.py",
        name=name,
        unit_type="function",
        start_line=1,
        end_line=5,
        source=source,
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """Returns a pre-set vector for a given query string -- same
    pattern as test_hybrid_ranker.py's FakeEmbedder."""

    def __init__(self, vector_by_query: dict):
        self._vector_by_query = vector_by_query

    def encode_one(self, text: str) -> np.ndarray:
        return self._vector_by_query[text]


def test_default_k_is_60():
    # 60 is the value from the original RRF paper (Cormack, Clarke &
    # Buettcher, 2009) and the de facto standard default.
    assert DEFAULT_RRF_K == 60


def test_invalid_k_raises():
    index = InvertedIndex()
    store = VectorStore()
    with pytest.raises(ValueError):
        RRFRanker(index, store, FakeEmbedder({}), k=0)


def test_negative_k_raises():
    index = InvertedIndex()
    store = VectorStore()
    with pytest.raises(ValueError):
        RRFRanker(index, store, FakeEmbedder({}), k=-5)


def test_search_on_empty_corpus_returns_empty_list():
    index = InvertedIndex()
    index.build([])
    store = VectorStore()
    store.build(unit_ids=[], embeddings=np.empty((0, 2)))

    ranker = RRFRanker(index, store, FakeEmbedder({"q": np.array([1.0, 0.0])}))
    assert ranker.search("q", top_k=5) == []


# Shared two-unit setup: unit 1 has a full keyword match (BM25 rank 1)
# but a poor semantic match (cosine rank 2, the worse of the two);
# unit 2 has zero keyword overlap (absent from BM25 entirely) but a
# perfect semantic match (cosine rank 1).
QUERY = "widget lookup"
UNITS = [
    _make_unit(1, "widget_lookup", "def widget_lookup(): pass"),
    _make_unit(2, "unrelated_thing", "def unrelated_thing(): pass"),
]
VECTORS = [_normalize(np.array([0.0, 1.0])), _normalize(np.array([1.0, 0.0]))]
QUERY_VECTOR = _normalize(np.array([1.0, 0.0]))


def _build_ranker(k=DEFAULT_RRF_K):
    index = InvertedIndex()
    index.build(UNITS)

    store = VectorStore()
    store.build(unit_ids=[u.id for u in UNITS], embeddings=np.array(VECTORS))

    embedder = FakeEmbedder({QUERY: QUERY_VECTOR})
    return RRFRanker(index, store, embedder, k=k)


def test_manual_score_calculation_matches_formula():
    k = 60
    ranker = _build_ranker(k=k)
    results = {r.unit_id: r for r in ranker.search(QUERY, top_k=2)}

    assert results[1].bm25_rank == 1
    assert results[1].cosine_rank == 2
    assert results[1].rrf_score == pytest.approx(1 / (k + 1) + 1 / (k + 2))

    assert results[2].bm25_rank is None
    assert results[2].cosine_rank == 1
    assert results[2].rrf_score == pytest.approx(1 / (k + 1))


def test_appearing_in_both_lists_beats_top_rank_in_only_one_list():
    # unit 1 appears in BOTH lists (rank 1 and rank 2); unit 2 has the
    # single best possible rank in one list (cosine rank 1) but is
    # completely absent from the other (zero keyword overlap). Two
    # modest contributions outscore one top contribution plus nothing
    # -- this is the entire point of rank fusion across two signals.
    ranker = _build_ranker()
    results = ranker.search(QUERY, top_k=2)

    assert results[0].unit_id == 1
    assert results[1].unit_id == 2
    assert results[0].rrf_score > results[1].rrf_score


def test_unit_missing_from_bm25_is_still_returned_via_cosine_alone():
    ranker = _build_ranker()
    results = {r.unit_id: r for r in ranker.search(QUERY, top_k=2)}

    assert 2 in results
    assert results[2].bm25_rank is None
    assert results[2].rrf_score > 0.0


def test_results_sorted_descending_by_rrf_score():
    ranker = _build_ranker()
    results = ranker.search(QUERY, top_k=2)
    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_number_of_results():
    ranker = _build_ranker()
    results = ranker.search(QUERY, top_k=1)
    assert len(results) == 1


def test_smaller_k_increases_relative_gap_between_ranks():
    # A smaller k makes the difference between rank 1 and rank 2
    # relatively larger (1/(k+1) vs 1/(k+2) diverge more as k shrinks),
    # since k dampens exactly that difference.
    small_k_ranker = _build_ranker(k=1)
    large_k_ranker = _build_ranker(k=1000)

    small_k_results = {r.unit_id: r.rrf_score for r in small_k_ranker.search(QUERY, top_k=2)}
    large_k_results = {r.unit_id: r.rrf_score for r in large_k_ranker.search(QUERY, top_k=2)}

    small_k_gap = small_k_results[1] - small_k_results[2]
    large_k_gap = large_k_results[1] - large_k_results[2]

    assert small_k_gap > large_k_gap
