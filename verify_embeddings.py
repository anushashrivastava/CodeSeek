"""Run this LOCALLY (not in a network-restricted sandbox) to verify
real semantic search quality against the sample repo, using the
actual pretrained model.

Usage:
    cd code-search-engine
    python3 verify_embeddings.py

First run will download all-MiniLM-L6-v2 (~80MB) from Hugging Face.
"""

from backend.embeddings.embedder import CodeEmbedder
from backend.pipeline import build_search_pipeline

# --- Rebuild the same CodeUnits and embeddings used in later stages ---
embedder = CodeEmbedder()
all_units, _index, store = build_search_pipeline("sample_repo", embedder)

print(f"Parsed {len(all_units)} code units from sample_repo.\n")

unit_by_id = {u.id: u for u in all_units}

# --- The queries that matter most: ones with LOW lexical overlap,
# to specifically test what keyword search structurally cannot do. ---
queries = [
    "where do we check if someone is logged in",       # vs is_session_expired
    "verify someone is who they claim to be",           # vs check_credentials
    "prevent someone from spamming an endpoint",         # vs rate_limit
    "make sure user input is clean before using it",     # vs sanitize_input
]

for q in queries:
    print(f'QUERY: "{q}"')
    query_vec = embedder.encode_one(q)
    results = store.search(query_vec, top_k=5)
    for unit_id, score in results:
        u = unit_by_id[unit_id]
        parent = f"{u.parent_class}." if u.parent_class else ""
        print(f"  {score:.3f}  {u.file_path:20s} {parent}{u.name}")
    print()
