"""Stores embedding vectors for CodeUnits and performs similarity search.

Design decision: brute-force search via a single matrix multiplication,
not an approximate-nearest-neighbor index (e.g. FAISS, HNSW).

Tradeoff, stated explicitly: brute force is O(N) per query — for
every search, we compare the query vector against all N stored
vectors. For a codebase of hundreds or a few thousand functions,
this is genuinely fast (a few milliseconds) and exact (not
approximate). A real vector database or ANN index earns its
complexity once N reaches the hundreds of thousands to millions —
at that point O(N) brute force becomes a real bottleneck and
approximate methods trade a small accuracy loss for large speed
gains. For this project's scale, introducing FAISS/a vector DB
would be complexity added for a problem we don't actually have.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Holds a matrix of unit-normalized embeddings and their unit ids."""

    def __init__(self):
        self._matrix: np.ndarray = np.empty((0, 0))
        self._unit_ids: List[int] = []

    def build(self, unit_ids: List[int], embeddings: np.ndarray) -> None:
        """Store embeddings alongside the unit ids they correspond to.

        embeddings must be shape (len(unit_ids), embedding_dim) and
        each row should already be unit-normalized (see CodeEmbedder,
        which sets normalize_embeddings=True) so that a plain dot
        product below equals cosine similarity.
        """
        if len(unit_ids) != embeddings.shape[0]:
            raise ValueError(
                f"unit_ids length ({len(unit_ids)}) must match "
                f"embeddings row count ({embeddings.shape[0]})"
            )
        self._matrix = embeddings
        self._unit_ids = list(unit_ids)
        logger.info(
            "Built vector store: %d vectors, dimension %d",
            self._matrix.shape[0],
            self._matrix.shape[1] if self._matrix.size else 0,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[tuple]:
        """Return up to top_k (unit_id, cosine_similarity) pairs, sorted
        by descending similarity.

        The core operation is one matrix-vector product: multiplying
        the (N, dim) stored matrix by the (dim,) query vector gives an
        (N,) array of similarity scores in a single vectorized NumPy
        call, rather than looping over N vectors in Python — this is
        what keeps brute-force search fast even for a few thousand units.
        """
        if self._matrix.size == 0:
            return []

        similarities = self._matrix @ query_vector  # shape: (N,)

        # argsort ascending, then take the last top_k and reverse them
        # to get descending order — avoids a full O(N log N) sort when
        # we only actually need the top few results, though for the
        # small N in this project a full sort would be fine too.
        top_k = min(top_k, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [(self._unit_ids[i], float(similarities[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self._unit_ids)
