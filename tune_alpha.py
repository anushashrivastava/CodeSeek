"""Follow-up to evaluate.py's Stage 8 finding: at the default
alpha=0.5, hybrid search underperforms semantic-only alone on the
30-query label set (mean Recall@5 0.933 vs. 0.967). That finding left
an open question -- is 0.5 just a bad choice of alpha, or is
min-max-normalized fusion unable to beat semantic-only at any weight
on this label set? This script answers it empirically rather than
guessing.

Sweeps alpha from 0.0 to 1.0 in steps of 0.1, reusing the exact same
pipeline and label set as evaluate.py (via build_pipeline() and
EVAL_QUERIES) so results are directly comparable -- only alpha varies.

Usage:
    python3 tune_alpha.py
"""

from backend.retrieval.hybrid_ranker import HybridRanker
from evaluate import EVAL_QUERIES, TOP_K, build_pipeline, precision_at_k, qualified_name, recall_at_k

ALPHA_STEPS = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0


def mean_precision_recall(ranker: HybridRanker, unit_by_id: dict) -> tuple:
    total_p = 0.0
    total_r = 0.0
    for entry in EVAL_QUERIES:
        relevant = set(entry["relevant"])
        results = ranker.search(entry["query"], top_k=TOP_K)
        ranked_names = [qualified_name(unit_by_id[r.unit_id]) for r in results]
        total_p += precision_at_k(ranked_names, relevant, TOP_K)
        total_r += recall_at_k(ranked_names, relevant, TOP_K)

    n = len(EVAL_QUERIES)
    return total_p / n, total_r / n


def main():
    unit_by_id, index, store, embedder = build_pipeline()

    print(f"Sweeping alpha over {ALPHA_STEPS} ({len(EVAL_QUERIES)} queries per value)...\n")

    rows = []
    for alpha in ALPHA_STEPS:
        ranker = HybridRanker(index, store, embedder, alpha=alpha)
        mean_p, mean_r = mean_precision_recall(ranker, unit_by_id)
        rows.append((alpha, mean_p, mean_r))
        print(f"  alpha={alpha:.1f}   mean P@{TOP_K}={mean_p:.3f}   mean R@{TOP_K}={mean_r:.3f}")

    best_by_recall = max(rows, key=lambda row: row[2])
    best_by_precision = max(rows, key=lambda row: row[1])

    print()
    print(f"Best mean Recall@{TOP_K}:    alpha={best_by_recall[0]:.1f} (R@{TOP_K}={best_by_recall[2]:.3f})")
    print(f"Best mean Precision@{TOP_K}: alpha={best_by_precision[0]:.1f} (P@{TOP_K}={best_by_precision[1]:.3f})")


if __name__ == "__main__":
    main()
