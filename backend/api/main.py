"""Thin FastAPI wrapper over the already-built retrieval pipeline.

Deliberately thin: this module contains no search, ranking, or
indexing logic of its own. It only (a) exposes POST /index, which
calls the shared build_search_pipeline() (see backend/pipeline.py),
and (b) exposes GET /search, which calls the already-tested
HybridRanker. Every hard problem (BM25, cosine similarity, score
fusion) was solved and verified in earlier stages before this file was
written.
"""

import logging
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.embeddings.embedder import CodeEmbedder
from backend.embeddings.vector_store import VectorStore
from backend.indexing.inverted_index import InvertedIndex
from backend.parser.models import CodeUnit
from backend.pipeline import build_search_pipeline
from backend.retrieval.hybrid_ranker import DEFAULT_ALPHA, HybridRanker

logger = logging.getLogger(__name__)

app = FastAPI(title="Code Search Engine")

# Wide-open CORS is a deliberate, scoped choice: this is a local,
# single-user dev tool (see README ground rules), not a multi-tenant
# service handling untrusted browsers or sensitive data. It exists
# purely so the plain HTML/JS frontend (served from a different local
# port) can call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexState:
    """Holds the currently-indexed repository's search structures.

    A single global, in-memory, most-recent-repo-wins state is a
    deliberate simplification, not an oversight: this project has no
    multi-repo or multi-user requirement, so there is exactly one
    "currently indexed repository" at a time -- the same mental model
    as running verify_hybrid_ranker.py by hand. Re-indexing replaces
    this state entirely rather than updating it incrementally;
    incremental indexing is out of scope for this project's timeline.
    """

    def __init__(self):
        self.repo_path: Optional[str] = None
        self.units_by_id: Dict[int, CodeUnit] = {}
        self.inverted_index: Optional[InvertedIndex] = None
        self.vector_store: Optional[VectorStore] = None

    @property
    def is_indexed(self) -> bool:
        return self.inverted_index is not None


_state = IndexState()
_embedder: Optional[CodeEmbedder] = None


def get_embedder() -> CodeEmbedder:
    """FastAPI dependency for the shared embedder instance.

    The model (~80MB) is loaded once, lazily, on first use, and reused
    for every request after that -- reloading it per-request would
    repeat an expensive model load for no benefit, since inference is
    stateless. Exposed as a dependency (not a bare module-level
    singleton) specifically so tests can override it via
    app.dependency_overrides[get_embedder] with a fake embedder,
    keeping API tests offline -- the same reasoning behind
    HybridRanker's QueryEmbedder Protocol.
    """
    global _embedder
    if _embedder is None:
        _embedder = CodeEmbedder()
    return _embedder


class IndexRequest(BaseModel):
    repo_path: str


class IndexResponse(BaseModel):
    repo_path: str
    indexed_units: int


class SearchResult(BaseModel):
    unit_id: int
    file_path: str
    name: str
    unit_type: str
    parent_class: Optional[str]
    start_line: int
    end_line: int
    snippet: str
    # Both raw and normalized scores for every component are returned,
    # not just the fused hybrid_score -- so a client (or a person
    # debugging a result) can see *why* something ranked where it did,
    # matching HybridResult's own explainability goal end to end.
    hybrid_score: float
    bm25_score: float
    bm25_normalized: float
    cosine_score: float
    cosine_normalized: float


class SearchResponse(BaseModel):
    query: str
    alpha: float
    results: List[SearchResult]


@app.get("/health")
def health():
    """Liveness check only -- deliberately does not report indexing
    status. /search already reports "nothing indexed yet" via its own
    409, so /health can answer the simpler, separate question "is the
    process up" without conflating the two."""
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse)
def index_repository(request: IndexRequest, embedder: CodeEmbedder = Depends(get_embedder)):
    """Index a repository -- reachable over HTTP. Replaces any
    previously indexed repository."""
    try:
        all_units, inverted_index, vector_store = build_search_pipeline(request.repo_path, embedder)
    except (FileNotFoundError, NotADirectoryError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    _state.repo_path = request.repo_path
    _state.units_by_id = {u.id: u for u in all_units}
    _state.inverted_index = inverted_index
    _state.vector_store = vector_store

    logger.info("Indexed %s: %d units", request.repo_path, len(all_units))
    return IndexResponse(repo_path=request.repo_path, indexed_units=len(all_units))


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    alpha: float = Query(DEFAULT_ALPHA, ge=0.0, le=1.0),
    embedder: CodeEmbedder = Depends(get_embedder),
):
    """Search the currently indexed repository.

    Requires a prior successful call to /index: a search against
    nothing indexed isn't a meaningful empty result, it's a client
    error (they forgot to index first), so it's a 409, not a 200 with
    an empty results list.

    top_k and alpha are bounded via FastAPI's own Query validation
    (ge/le) rather than manual checks, since alpha outside [0, 1] or a
    non-positive top_k are malformed requests, not empty-result cases.
    """
    if not _state.is_indexed:
        raise HTTPException(status_code=409, detail="No repository indexed yet. Call POST /index first.")

    ranker = HybridRanker(_state.inverted_index, _state.vector_store, embedder, alpha=alpha)
    results = ranker.search(q, top_k=top_k)

    response_results = []
    for r in results:
        unit = _state.units_by_id[r.unit_id]
        response_results.append(
            SearchResult(
                unit_id=r.unit_id,
                file_path=unit.file_path,
                name=unit.name,
                unit_type=unit.unit_type,
                parent_class=unit.parent_class,
                start_line=unit.start_line,
                end_line=unit.end_line,
                snippet=unit.snippet(),
                hybrid_score=r.hybrid_score,
                bm25_score=r.bm25_score,
                bm25_normalized=r.bm25_normalized,
                cosine_score=r.cosine_score,
                cosine_normalized=r.cosine_normalized,
            )
        )

    return SearchResponse(query=q, alpha=alpha, results=response_results)
