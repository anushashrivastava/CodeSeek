"""Shared indexing pipeline: scan a repository, parse it into
CodeUnits, and build both retrieval structures (BM25 index + embedding
store).

Factored out because this exact sequence -- scan_repository ->
parse_file per file -> InvertedIndex.build -> embed searchable_text ->
VectorStore.build -- was duplicated near-identically across
backend/api/main.py, evaluate.py, verify_embeddings.py, and
verify_hybrid_ranker.py. A single shared implementation means the one
place that actually defines "how a repository gets indexed" is this
file, not four near-copies that could silently drift apart.

The embedder is a parameter, not constructed here, so callers that
need dependency injection (the FastAPI app's test overrides; a
CLI script's real CodeEmbedder) keep control over which one is used.
"""

from typing import List, Tuple

from backend.embeddings.vector_store import VectorStore
from backend.indexing.inverted_index import InvertedIndex
from backend.parser.ast_parser import parse_file
from backend.parser.models import CodeUnit
from backend.parser.scanner import scan_repository, to_relative_path


def build_search_pipeline(repo_path: str, embedder) -> Tuple[List[CodeUnit], InvertedIndex, VectorStore]:
    """Scan, parse, and index a repository. Returns (all_units, index, store).

    `embedder` needs only an `encode(texts) -> np.ndarray` method (see
    CodeEmbedder) -- not typed as a Protocol here since, unlike
    HybridRanker/RRFRanker's per-query encode_one(), this is the one
    place batch-encoding happens, which only CodeEmbedder itself and
    test fakes built for this specific call site need to implement.

    Raises FileNotFoundError/NotADirectoryError (from scan_repository)
    for a bad repo_path -- callers that need to turn that into an HTTP
    error (see backend/api/main.py) catch it themselves; this function
    stays a plain library call with no HTTP awareness.
    """
    files = scan_repository(repo_path)
    all_units: List[CodeUnit] = []
    next_id = 0
    for f in sorted(files):
        rel = to_relative_path(f, repo_path)
        units = parse_file(f, rel, next_id)
        all_units.extend(units)
        next_id += len(units)

    index = InvertedIndex()
    index.build(all_units)

    store = VectorStore()
    if all_units:
        texts = [u.searchable_text() for u in all_units]
        embeddings = embedder.encode(texts)
        store.build(unit_ids=[u.id for u in all_units], embeddings=embeddings)

    return all_units, index, store
