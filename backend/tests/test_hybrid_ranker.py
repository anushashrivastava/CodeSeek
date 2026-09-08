"""Unit tests for HybridRanker: BM25 + cosine fusion via min-max normalization.

Uses a fake embedder (no real sentence-transformers model) so these
tests run offline and fast, the same way test_vector_store.py tests
cosine search mechanics without a real model. What's under test here
is the *fusion math* (normalization + weighting) — real semantic
quality is a separate concern already validated with the real model in
verify_embeddings.py.
"""

import numpy as np
import pytest

from backend.embeddings.vector_store import VectorStore
from backend.indexing.inverted_index import InvertedIndex
from backend.parser.models import CodeUnit
from backend.retrieval.hybrid_ranker import HybridRanker, _min_max_normalize


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
    """Returns a pre-set vector for a given query string, so tests
    fully control the "semantic" side of the fusion without loading a
    real model."""

    def __init__(self, vector_by_query: dict):
        self._vector_by_query = vector_by_query

    def encode_one(self, text: str) -> np.ndarray:
        return self._vector_by_query[text]


def _build_ranker(units, unit_vectors, query, query_vector, alpha=0.5):
    index = InvertedIndex()
    index.build(units)

    store = VectorStore()
    store.build(unit_ids=[u.id for u in units], embeddings=np.array(unit_vectors))

    embedder = FakeEmbedder({query: query_vector})
    return HybridRanker(index, store, embedder, alpha=alpha)


# Shared fixture-like setup used by most tests below: two units where
# keyword overlap and semantic similarity deliberately point at
# *different* units, so alpha's effect on ranking is unambiguous.
#
# unit 1 "unrelated_thing": zero lexical overlap with the query, but
#   its vector is set to perfectly match the query vector (cosine=1.0).
# unit 2 "widget_lookup": full lexical overlap with the query
#   ("widget lookup"), but its vector is orthogonal to the query
#   (cosine=0.0).
QUERY = "widget lookup"
UNITS = [
    _make_unit(1, "unrelated_thing", "def unrelated_thing(): pass"),
    _make_unit(2, "widget_lookup", "def widget_lookup(): pass"),
]
VECTORS = [_normalize(np.array([1.0, 0.0])), _normalize(np.array([0.0, 1.0]))]
QUERY_VECTOR = _normalize(np.array([1.0, 0.0]))


def test_pure_semantic_alpha_ignores_keyword_overlap():
    ranker = _build_ranker(UNITS, VECTORS, QUERY, QUERY_VECTOR, alpha=1.0)
    results = ranker.search(QUERY, top_k=2)

    assert results[0].unit_id == 1
    assert results[0].hybrid_score == pytest.approx(1.0, abs=1e-6)
    assert results[1].hybrid_score == pytest.approx(0.0, abs=1e-6)


def test_pure_keyword_alpha_ignores_semantic_similarity():
    ranker = _build_ranker(UNITS, VECTORS, QUERY, QUERY_VECTOR, alpha=0.0)
    results = ranker.search(QUERY, top_k=2)

    assert results[0].unit_id == 2
    assert results[0].hybrid_score == pytest.approx(1.0, abs=1e-6)
    assert results[1].hybrid_score == pytest.approx(0.0, abs=1e-6)


def test_alpha_weight_shifts_ranking_between_keyword_and_semantic():
    # alpha=0.3 weighs keyword (bm25) more heavily than semantic:
    # unit 1 (semantic winner): 0.3*1.0 + 0.7*0.0 = 0.3
    # unit 2 (keyword winner):  0.3*0.0 + 0.7*1.0 = 0.7
    ranker = _build_ranker(UNITS, VECTORS, QUERY, QUERY_VECTOR, alpha=0.3)
    results = ranker.search(QUERY, top_k=2)

    assert results[0].unit_id == 2
    assert results[0].hybrid_score == pytest.approx(0.7, abs=1e-6)
    assert results[1].unit_id == 1
    assert results[1].hybrid_score == pytest.approx(0.3, abs=1e-6)


def test_equal_alpha_produces_tied_scores_for_symmetric_case():
    ranker = _build_ranker(UNITS, VECTORS, QUERY, QUERY_VECTOR, alpha=0.5)
    results = ranker.search(QUERY, top_k=2)

    assert results[0].hybrid_score == pytest.approx(0.5, abs=1e-6)
    assert results[1].hybrid_score == pytest.approx(0.5, abs=1e-6)


def test_zero_keyword_overlap_unit_still_appears_with_zero_bm25_score():
    ranker = _build_ranker(UNITS, VECTORS, QUERY, QUERY_VECTOR, alpha=0.5)
    results = {r.unit_id: r for r in ranker.search(QUERY, top_k=2)}

    assert results[1].bm25_score == 0.0
    assert results[1].bm25_normalized == 0.0


def test_result_exposes_raw_and_normalized_component_scores():
    ranker = _build_ranker(UNITS, VECTORS, QUERY, QUERY_VECTOR, alpha=0.5)
    result = ranker.search(QUERY, top_k=1)[0]

    assert hasattr(result, "bm25_score")
    assert hasattr(result, "bm25_normalized")
    assert hasattr(result, "cosine_score")
    assert hasattr(result, "cosine_normalized")
    assert hasattr(result, "hybrid_score")


def test_top_k_limits_number_of_hybrid_results():
    ranker = _build_ranker(UNITS, VECTORS, QUERY, QUERY_VECTOR, alpha=0.5)
    results = ranker.search(QUERY, top_k=1)
    assert len(results) == 1


def test_invalid_alpha_raises():
    index = InvertedIndex()
    store = VectorStore()
    with pytest.raises(ValueError):
        HybridRanker(index, store, FakeEmbedder({}), alpha=1.5)


def test_search_on_empty_corpus_returns_empty_list():
    index = InvertedIndex()
    index.build([])
    store = VectorStore()
    store.build(unit_ids=[], embeddings=np.empty((0, 2)))

    ranker = HybridRanker(index, store, FakeEmbedder({"q": np.array([1.0, 0.0])}), alpha=0.5)
    assert ranker.search("q", top_k=5) == []


# ---------- _min_max_normalize edge cases ----------

def test_min_max_normalize_rescales_to_zero_one_range():
    result = _min_max_normalize({1: 10.0, 2: 20.0, 3: 30.0})
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(0.5)
    assert result[3] == pytest.approx(1.0)


def test_min_max_normalize_handles_identical_scores_without_dividing_by_zero():
    result = _min_max_normalize({1: 5.0, 2: 5.0})
    assert result == {1: 0.0, 2: 0.0}


def test_min_max_normalize_handles_empty_input():
    assert _min_max_normalize({}) == {}
