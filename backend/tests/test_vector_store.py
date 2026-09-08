"""Unit tests for VectorStore.

These use small, hand-constructed vectors rather than a real embedding
model, so they run fast, need no network access, and test the
similarity-search *mechanics* precisely (we know exactly what the
"correct" answer should be, since we built the vectors ourselves).

Real semantic quality (does the model actually understand meaning) is
a separate concern validated later with the real model, not here.
"""

import numpy as np
import pytest

from backend.embeddings.vector_store import VectorStore


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_identical_vector_scores_highest():
    vectors = np.array([
        normalize(np.array([1.0, 0.0, 0.0])),
        normalize(np.array([0.0, 1.0, 0.0])),
        normalize(np.array([0.9, 0.1, 0.0])),
    ])
    store = VectorStore()
    store.build(unit_ids=[10, 20, 30], embeddings=vectors)

    query = normalize(np.array([1.0, 0.0, 0.0]))
    results = store.search(query, top_k=3)

    assert results[0][0] == 10
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_vector_scores_near_zero():
    vectors = np.array([
        normalize(np.array([1.0, 0.0])),
        normalize(np.array([0.0, 1.0])),
    ])
    store = VectorStore()
    store.build(unit_ids=[1, 2], embeddings=vectors)

    query = normalize(np.array([1.0, 0.0]))
    results = dict(store.search(query, top_k=2))

    assert results[1] == pytest.approx(1.0, abs=1e-6)
    assert results[2] == pytest.approx(0.0, abs=1e-6)


def test_opposite_vector_scores_negative_one():
    vectors = np.array([normalize(np.array([1.0, 0.0]))])
    store = VectorStore()
    store.build(unit_ids=[1], embeddings=vectors)

    query = normalize(np.array([-1.0, 0.0]))
    results = store.search(query, top_k=1)

    assert results[0][1] == pytest.approx(-1.0, abs=1e-6)


def test_results_sorted_descending_by_similarity():
    vectors = np.array([
        normalize(np.array([1.0, 0.0])),
        normalize(np.array([0.7, 0.7])),
        normalize(np.array([0.0, 1.0])),
    ])
    store = VectorStore()
    store.build(unit_ids=[1, 2, 3], embeddings=vectors)

    query = normalize(np.array([1.0, 0.0]))
    results = store.search(query, top_k=3)
    scores = [score for _, score in results]

    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_number_of_results():
    vectors = np.array([normalize(np.random.rand(5)) for _ in range(10)])
    store = VectorStore()
    store.build(unit_ids=list(range(10)), embeddings=vectors)

    results = store.search(normalize(np.random.rand(5)), top_k=3)
    assert len(results) == 3


def test_search_on_empty_store_returns_empty_list():
    store = VectorStore()
    results = store.search(np.array([1.0, 0.0]), top_k=5)
    assert results == []


def test_build_raises_on_mismatched_lengths():
    store = VectorStore()
    with pytest.raises(ValueError):
        store.build(unit_ids=[1, 2, 3], embeddings=np.zeros((2, 4)))


def test_len_matches_number_of_stored_vectors():
    vectors = np.array([normalize(np.random.rand(4)) for _ in range(7)])
    store = VectorStore()
    store.build(unit_ids=list(range(7)), embeddings=vectors)
    assert len(store) == 7
