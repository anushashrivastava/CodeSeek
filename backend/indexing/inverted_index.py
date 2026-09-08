"""Inverted index over CodeUnits, with BM25 ranking.

The index maps each term to the set of units containing it and how
often. BM25 uses that plus corpus-wide statistics (document frequency,
average document length) to rank units for a given query.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List

from backend.indexing.tokenizer import tokenize
from backend.parser.models import CodeUnit

logger = logging.getLogger(__name__)

# Standard BM25 defaults used by Lucene/Elasticsearch. k1 controls
# term-frequency saturation (higher = extra occurrences keep mattering
# longer); b controls how strongly document length is penalized
# (0 = no length penalty at all, 1 = full proportional penalty).
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class InvertedIndex:
    """Builds and queries a BM25-ranked inverted index over CodeUnits."""

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        self.k1 = k1
        self.b = b

        # term -> {unit_id: term_frequency_in_that_unit}
        self._postings: Dict[str, Dict[int, int]] = defaultdict(dict)

        # unit_id -> total token count in that unit (its "document length")
        self._doc_lengths: Dict[int, int] = {}

        # unit_id -> CodeUnit, so search results can return full objects
        self._units: Dict[int, CodeUnit] = {}

        self._avg_doc_length: float = 0.0

    def build(self, units: List[CodeUnit]) -> None:
        """Build the index from scratch given a list of CodeUnits.

        We tokenize each unit's searchable_text() (name + docstring +
        source, see CodeUnit.searchable_text) and record term
        frequencies. This is a one-time cost paid at indexing time,
        not at query time.
        """
        self._postings.clear()
        self._doc_lengths.clear()
        self._units.clear()

        for unit in units:
            self._units[unit.id] = unit
            tokens = tokenize(unit.searchable_text())
            self._doc_lengths[unit.id] = len(tokens)

            term_counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1

            for term, count in term_counts.items():
                self._postings[term][unit.id] = count

        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = total_length / len(units) if units else 0.0

        logger.info(
            "Built inverted index: %d units, %d unique terms, avg doc length %.1f",
            len(units), len(self._postings), self._avg_doc_length,
        )

    def _idf(self, term: str) -> float:
        """Inverse document frequency for a term, using the standard
        BM25 IDF formula (a smoothed variant that stays positive for
        common terms rather than going negative, as the classic
        textbook IDF formula can for terms in more than half the corpus).
        """
        n = len(self._units)
        df = len(self._postings.get(term, {}))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Return up to top_k (unit_id, bm25_score) pairs for a query,
        sorted by descending score. Units that share zero query terms
        with the query are never scored or returned — BM25 only ranks
        among documents that have at least some lexical overlap.
        """
        if not self._units:
            return []

        query_terms = tokenize(query)
        scores: Dict[int, float] = defaultdict(float)

        for term in query_terms:
            postings = self._postings.get(term)
            if not postings:
                continue  # term never appears in the corpus at all

            idf = self._idf(term)

            for unit_id, tf in postings.items():
                doc_len = self._doc_lengths[unit_id]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_length)
                scores[unit_id] += idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

    def get_unit(self, unit_id: int) -> CodeUnit:
        """Look up the full CodeUnit for a given id."""
        return self._units[unit_id]

    def __len__(self) -> int:
        return len(self._units)
