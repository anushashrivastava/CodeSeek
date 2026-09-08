"""Reciprocal Rank Fusion (RRF): an alternative fusion strategy to
HybridRanker's min-max score normalization, evaluated side by side
with it rather than replacing it.

Why this exists alongside HybridRanker, not instead of it: Stage 8's
evaluation (see README.md) found a real weakness in min-max score
fusion -- a unit with strong *incidental* keyword overlap can, once
normalized, still outrank a genuinely relevant unit with a real but
modest semantic score (the is_session_expired regression). RRF fuses
*rank positions* instead of scores, so it never has to normalize two
differently-scaled, differently-distributed score sets against each
other in the first place -- it doesn't care whether BM25's scores are
0-20 and cosine's are 0.1-0.5, only about each unit's position in each
ranked list. Whether that actually makes it better on this project's
data is an empirical question, answered for real in evaluate.py /
README.md's "Stage 9" section -- not assumed here.

RRF formula (Cormack, Clarke & Buettcher, 2009 -- the paper that
introduced it for combining search engine result lists):

    RRF_score(unit) = sum over each ranked list L containing unit of
                      1 / (k + rank_L(unit))

where rank_L(unit) is the unit's 1-indexed position in list L, and k
is a constant (60, the value used in the original paper and the value
most implementations default to) that dampens how much a #1 ranking in
one list can dominate the fusion outright.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from backend.embeddings.vector_store import VectorStore
from backend.indexing.inverted_index import InvertedIndex
from backend.retrieval.hybrid_ranker import QueryEmbedder

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


@dataclass
class RRFResult:
    """One ranked result. Unlike HybridResult, there is no "normalized
    score" to show -- RRF's entire premise is that it never computes
    one -- so what's exposed instead, for the same explainability
    reason, is each method's raw rank (None if the unit never appeared
    in that method's ranking at all) alongside the final fused score.
    """

    unit_id: int
    rrf_score: float
    bm25_rank: Optional[int]
    cosine_rank: Optional[int]


class RRFRanker:
    """Fuses BM25 and cosine-similarity RANKINGS (not scores) via
    Reciprocal Rank Fusion. Same constructor shape as HybridRanker
    (inverted_index, vector_store, embedder) so the two are
    interchangeable in evaluation code -- only the fusion strategy
    differs.
    """

    def __init__(
        self,
        inverted_index: InvertedIndex,
        vector_store: VectorStore,
        embedder: QueryEmbedder,
        k: int = DEFAULT_RRF_K,
    ):
        """k dampens the influence of top ranks: a smaller k makes rank
        1 vs. rank 2 a bigger relative difference; a larger k flattens
        that difference out. 60 is the standard default from the
        original paper and is not tuned here -- tuning k is the RRF
        analogue of tuning HybridRanker's alpha, and is explicitly out
        of scope for this comparison (see README.md's "Stage 9").
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        self.inverted_index = inverted_index
        self.vector_store = vector_store
        self.embedder = embedder
        self.k = k

    def search(self, query: str, top_k: int = 10) -> List[RRFResult]:
        """Return up to top_k results ranked by fused RRF score.

        Both underlying searches are run exhaustively (top_k = full
        corpus size) so every unit's true rank in each list is known,
        not just whichever units happened to land in each method's own
        top-10 -- the same reasoning HybridRanker.search() uses.
        """
        corpus_size = len(self.inverted_index)
        if corpus_size == 0:
            return []

        bm25_ranked = self.inverted_index.search(query, top_k=corpus_size)
        query_vector = self.embedder.encode_one(query)
        cosine_ranked = self.vector_store.search(query_vector, top_k=corpus_size)

        # 1-indexed rank position in each list. BM25 only returns units
        # sharing at least one query term, so a unit with zero lexical
        # overlap simply has no entry here -- unlike HybridRanker, RRF
        # does not invent a rank for it; it just contributes nothing
        # from this list, which is exactly what "not ranked by this
        # method" should mean under RRF.
        bm25_rank_by_id = {unit_id: rank for rank, (unit_id, _) in enumerate(bm25_ranked, start=1)}
        cosine_rank_by_id = {unit_id: rank for rank, (unit_id, _) in enumerate(cosine_ranked, start=1)}

        # Cosine similarity is computed for every stored vector, so
        # cosine_rank_by_id always covers the full corpus; the union
        # with bm25_rank_by_id is included anyway so nothing could be
        # silently dropped if that ever changed.
        all_unit_ids = set(bm25_rank_by_id) | set(cosine_rank_by_id)

        results = []
        for unit_id in all_unit_ids:
            bm25_rank = bm25_rank_by_id.get(unit_id)
            cosine_rank = cosine_rank_by_id.get(unit_id)

            rrf_score = 0.0
            if bm25_rank is not None:
                rrf_score += 1.0 / (self.k + bm25_rank)
            if cosine_rank is not None:
                rrf_score += 1.0 / (self.k + cosine_rank)

            results.append(
                RRFResult(unit_id=unit_id, rrf_score=rrf_score, bm25_rank=bm25_rank, cosine_rank=cosine_rank)
            )

        results.sort(key=lambda r: r.rrf_score, reverse=True)
        return results[:top_k]
