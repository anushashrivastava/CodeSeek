"""Combines BM25 keyword search and cosine-similarity semantic search
into a single ranked result list.

Why this needs to exist at all: BM25 and cosine similarity are computed
by two completely independent systems (InvertedIndex, VectorStore) and
live on very different, incompatible scales. Confirmed with real data
in verify_embeddings.py: cosine similarities against sample_repo/ sit
in a narrow ~0.19-0.42 band, while BM25 is unbounded and typically much
larger for exact keyword matches. Averaging the two raw scores directly
would let whichever score happens to have the larger raw range dominate
the fusion regardless of actual relevance. Per-query min-max
normalization (rescaling each method's scores to [0, 1] before
combining) is the fix, and it has to be done per query because each
method's raw score range shifts depending on the query and corpus.
"""

import logging
from dataclasses import dataclass
from typing import List, Protocol

from backend.embeddings.vector_store import VectorStore
from backend.indexing.inverted_index import InvertedIndex

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.5


class QueryEmbedder(Protocol):
    """Anything that can turn a query string into a vector.

    Defined as a Protocol (structural typing) rather than importing
    CodeEmbedder directly, so tests can inject a fake embedder that
    returns hand-built vectors instead of loading the real
    sentence-transformers model. This keeps hybrid ranker tests
    offline and fast, matching the rest of the test suite.
    """

    def encode_one(self, text: str):
        ...


@dataclass
class HybridResult:
    """One ranked result, keeping both raw and normalized component
    scores visible rather than collapsing straight to a single number.

    Exposing the breakdown (not just the final hybrid_score) is
    deliberate: it's what makes it possible to explain, for any given
    result, *why* it ranked where it did — e.g. "this scored high
    because of keyword overlap, not semantic similarity" — which
    matters both for debugging and for the interview-explainability
    goal stated throughout this project.
    """

    unit_id: int
    hybrid_score: float
    bm25_score: float
    bm25_normalized: float
    cosine_score: float
    cosine_normalized: float


def _min_max_normalize(scores: dict) -> dict:
    """Rescale a {unit_id: raw_score} dict to [0, 1] via min-max.

    If every score is identical (including the common case of a
    single-unit corpus, or a query that matches nothing so every raw
    score is 0), min-max normalization has a zero denominator. In that
    case every unit is equally (ir)relevant by this method, so they
    all get 0.0 rather than dividing by zero.
    """
    if not scores:
        return {}

    lo = min(scores.values())
    hi = max(scores.values())
    spread = hi - lo

    if spread == 0:
        return {unit_id: 0.0 for unit_id in scores}

    return {unit_id: (score - lo) / spread for unit_id, score in scores.items()}


class HybridRanker:
    """Fuses BM25 and cosine-similarity rankings with a configurable weight."""

    def __init__(
        self,
        inverted_index: InvertedIndex,
        vector_store: VectorStore,
        embedder: QueryEmbedder,
        alpha: float = DEFAULT_ALPHA,
    ):
        """alpha controls how much weight semantic search gets:
        hybrid_score = alpha * cosine_normalized + (1 - alpha) * bm25_normalized.
        alpha=0.5 (default) weighs both equally; alpha=1.0 is
        semantic-only, alpha=0.0 is keyword-only. Exposed as a
        constructor argument, not hardcoded, because the right balance
        is an empirical question the Stage 8 evaluation script is
        meant to answer, not something to guess up front.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

        self.inverted_index = inverted_index
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha

    def search(self, query: str, top_k: int = 10) -> List[HybridResult]:
        """Return up to top_k results ranked by fused hybrid score.

        Both underlying searches are run exhaustively (top_k = full
        corpus size), not just for the final top_k, so every unit gets
        a real score from both methods before normalization. Merging
        two separately-truncated top-10 lists would risk silently
        dropping a unit that one method ranked highly but the other
        happened to place just outside its own top 10.
        """
        corpus_size = len(self.inverted_index)
        if corpus_size == 0:
            return []

        bm25_raw = dict(self.inverted_index.search(query, top_k=corpus_size))
        query_vector = self.embedder.encode_one(query)
        cosine_raw = dict(self.vector_store.search(query_vector, top_k=corpus_size))

        # BM25 only scores units sharing at least one query term, so
        # units with zero lexical overlap are simply absent from
        # bm25_raw. That absence itself is real signal (zero keyword
        # relevance), so we fill it in as an explicit 0 rather than
        # excluding those units from the fused ranking entirely.
        all_unit_ids = set(bm25_raw) | set(cosine_raw)
        bm25_raw = {uid: bm25_raw.get(uid, 0.0) for uid in all_unit_ids}
        cosine_raw = {uid: cosine_raw.get(uid, 0.0) for uid in all_unit_ids}

        bm25_norm = _min_max_normalize(bm25_raw)
        cosine_norm = _min_max_normalize(cosine_raw)

        results = [
            HybridResult(
                unit_id=uid,
                hybrid_score=self.alpha * cosine_norm[uid] + (1 - self.alpha) * bm25_norm[uid],
                bm25_score=bm25_raw[uid],
                bm25_normalized=bm25_norm[uid],
                cosine_score=cosine_raw[uid],
                cosine_normalized=cosine_norm[uid],
            )
            for uid in all_unit_ids
        ]

        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results[:top_k]
