"""Run this locally to see real hybrid ranking output against
sample_repo/, using the actual pretrained embedding model plus the
real BM25 inverted index.

Uses the same target queries as verify_embeddings.py so the hybrid
results can be directly compared against the semantic-only results
already recorded in README.md.

Usage:
    python3 verify_hybrid_ranker.py
"""

from backend.embeddings.embedder import CodeEmbedder
from backend.pipeline import build_search_pipeline
from backend.retrieval.hybrid_ranker import HybridRanker

# --- Rebuild the same CodeUnits and both retrieval systems used in
# later stages ---
embedder = CodeEmbedder()
all_units, index, store = build_search_pipeline("sample_repo", embedder)

print(f"Parsed {len(all_units)} code units from sample_repo.\n")

unit_by_id = {u.id: u for u in all_units}

# Same low-lexical-overlap queries used in verify_embeddings.py, so
# results are directly comparable to the semantic-only run recorded
# in README.md.
queries = [
    "where do we check if someone is logged in",
    "verify someone is who they claim to be",
    "prevent someone from spamming an endpoint",
    "make sure user input is clean before using it",
]

for alpha in (0.5,):
    ranker = HybridRanker(index, store, embedder, alpha=alpha)
    print(f"=== alpha={alpha} ===\n")
    for q in queries:
        print(f'QUERY: "{q}"')
        for r in ranker.search(q, top_k=5):
            u = unit_by_id[r.unit_id]
            parent = f"{u.parent_class}." if u.parent_class else ""
            print(
                f"  hybrid={r.hybrid_score:.3f}  "
                f"(bm25={r.bm25_score:.2f}->{r.bm25_normalized:.2f}, "
                f"cos={r.cosine_score:.3f}->{r.cosine_normalized:.2f})  "
                f"{u.file_path:20s} {parent}{u.name}"
            )
        print()
