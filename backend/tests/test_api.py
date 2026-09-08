"""Integration tests for the FastAPI wrapper.

Uses FastAPI's TestClient plus a fake embedder, dependency-injected via
app.dependency_overrides -- the same offline-testing approach as
test_hybrid_ranker.py: no real model, no network. A small synthetic
repo written to tmp_path stands in for a real repository, matching the
pattern already used in test_ast_parser.py, rather than depending on
sample_repo/'s exact contents.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app, get_embedder, _state


class FakeEmbedder:
    """Deterministic, hash-based fake embeddings.

    Enough to exercise the API's plumbing (shapes, status codes,
    response fields) without needing the real sentence-transformers
    model. Real embedding quality is a separate, already-verified
    concern (see verify_embeddings.py / verify_hybrid_ranker.py).
    """

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


@pytest.fixture(autouse=True)
def reset_state():
    """_state is a module-level singleton in backend.api.main (see its
    docstring for why), so it persists across tests in this file unless
    explicitly reset. Without this, test order would silently matter --
    e.g. test_search_before_indexing_returns_409 would only pass by
    accident of running before any test that indexes a repo."""
    _state.repo_path = None
    _state.units_by_id = {}
    _state.inverted_index = None
    _state.vector_store = None
    yield


@pytest.fixture
def client():
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def check_credentials(username, password):\n"
        "    '''Verify a username/password pair.'''\n"
        "    return username == 'admin'\n"
    )
    (tmp_path / "utils.py").write_text(
        "def sanitize_input(text):\n"
        "    '''Strip dangerous characters from user input.'''\n"
        "    return text.strip()\n"
    )
    return str(tmp_path)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_before_indexing_returns_409(client):
    response = client.get("/search", params={"q": "credentials"})
    assert response.status_code == 409


def test_index_returns_unit_count(client, sample_repo):
    response = client.post("/index", json={"repo_path": sample_repo})

    assert response.status_code == 200
    body = response.json()
    assert body["repo_path"] == sample_repo
    assert body["indexed_units"] == 2


def test_index_nonexistent_path_returns_400(client):
    response = client.post("/index", json={"repo_path": "C:/definitely/not/a/real/path"})
    assert response.status_code == 400


def test_search_after_indexing_finds_expected_unit(client, sample_repo):
    client.post("/index", json={"repo_path": sample_repo})
    response = client.get("/search", params={"q": "credentials", "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "credentials"
    names = [r["name"] for r in body["results"]]
    assert "check_credentials" in names


def test_search_result_exposes_full_score_breakdown(client, sample_repo):
    client.post("/index", json={"repo_path": sample_repo})
    response = client.get("/search", params={"q": "credentials"})
    result = response.json()["results"][0]

    for field in ("hybrid_score", "bm25_score", "bm25_normalized", "cosine_score", "cosine_normalized"):
        assert field in result


def test_search_rejects_out_of_range_alpha(client, sample_repo):
    client.post("/index", json={"repo_path": sample_repo})
    response = client.get("/search", params={"q": "credentials", "alpha": 1.5})
    assert response.status_code == 422


def test_search_rejects_non_positive_top_k(client, sample_repo):
    client.post("/index", json={"repo_path": sample_repo})
    response = client.get("/search", params={"q": "credentials", "top_k": 0})
    assert response.status_code == 422


def test_reindexing_replaces_previous_repo(client, sample_repo, tmp_path):
    client.post("/index", json={"repo_path": sample_repo})

    other_repo = tmp_path / "other"
    other_repo.mkdir()
    (other_repo / "only.py").write_text("def only_function():\n    pass\n")

    response = client.post("/index", json={"repo_path": str(other_repo)})

    assert response.status_code == 200
    assert response.json()["indexed_units"] == 1
