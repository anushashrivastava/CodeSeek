"""End-to-end integration test for backend/pipeline.py's
build_search_pipeline(): repository -> AST parsing -> indexing
(BM25 + embeddings) -> search, all in one real (non-HTTP) chain.

Complements test_api.py, which covers the same chain through the HTTP
layer (repository -> ... -> API response). This file is the dedicated
coverage for backend/pipeline.py itself -- before this file existed,
that module was only exercised indirectly, through the API tests and
the verify_*.py / evaluate.py scripts.
"""

import numpy as np
import pytest

from backend.pipeline import build_search_pipeline
from backend.retrieval.hybrid_ranker import HybridRanker


class FakeEmbedder:
    """Deterministic fake embeddings -- same approach as test_api.py's
    and test_hybrid_ranker.py's fakes, so this test needs no real model."""

    DIM = 8

    def _vector(self, text: str) -> np.ndarray:
        seed = sum(ord(c) for c in text) % (2**31)
        rng = np.random.RandomState(seed)
        v = rng.rand(self.DIM)
        return v / np.linalg.norm(v)

    def encode(self, texts):
        return np.array([self._vector(t) for t in texts])

    def encode_one(self, text: str) -> np.ndarray:
        return self._vector(text)


@pytest.fixture
def synthetic_repo(tmp_path):
    """A small multi-file repo touching a function, a class, and a
    method -- enough to exercise parsing, keyword indexing, and
    embedding together, without depending on sample_repo/'s contents."""
    (tmp_path / "auth.py").write_text(
        "def check_credentials(username, password):\n"
        "    '''Verify a username and password pair.'''\n"
        "    return username == 'admin' and password == 'secret'\n"
    )
    (tmp_path / "accounts.py").write_text(
        "class AccountManager:\n"
        "    '''Manages user account lifecycle.'''\n"
        "\n"
        "    def deactivate(self, username):\n"
        "        '''Mark an account inactive.'''\n"
        "        return {'username': username, 'active': False}\n"
    )
    return str(tmp_path)


def test_repository_to_search_end_to_end(synthetic_repo):
    """Walks the full chain: a real repository on disk -> scanner finds
    the files -> AST parser extracts CodeUnits -> InvertedIndex and
    VectorStore are built from them -> HybridRanker finds the right
    unit for a keyword-exact query."""
    embedder = FakeEmbedder()
    all_units, index, store = build_search_pipeline(synthetic_repo, embedder)

    # Parsing: exactly 3 units -- the function, the class, and its method.
    assert len(all_units) == 3
    assert {u.name for u in all_units} == {"check_credentials", "AccountManager", "deactivate"}

    # Indexing: both retrieval structures actually got built from those units.
    assert len(index) == 3
    assert len(store) == 3

    # Search: a keyword-exact query finds its unit via a HybridRanker
    # built directly on top of this pipeline's output. Uses alpha=0.0
    # (keyword-only) deliberately: FakeEmbedder's hash-based vectors
    # carry no real semantic meaning, so blending in cosine at the
    # default alpha would let arbitrary hash noise decide the top
    # result instead of the real BM25 signal this test means to check.
    unit_by_id = {u.id: u for u in all_units}
    ranker = HybridRanker(index, store, embedder, alpha=0.0)
    results = ranker.search("check_credentials", top_k=3)

    assert results[0].bm25_score > 0
    assert unit_by_id[results[0].unit_id].name == "check_credentials"


def test_empty_repository_produces_empty_but_queryable_pipeline(tmp_path):
    """A repository with no Python files should not crash the
    pipeline -- it should produce a valid, empty index/store that a
    ranker can still safely query."""
    embedder = FakeEmbedder()
    all_units, index, store = build_search_pipeline(str(tmp_path), embedder)

    assert all_units == []
    assert len(index) == 0
    assert len(store) == 0

    ranker = HybridRanker(index, store, embedder)
    assert ranker.search("anything", top_k=5) == []
