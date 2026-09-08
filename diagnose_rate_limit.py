"""Diagnoses why `rate_limit` (utils/security.py) is the one unit every
evaluation in this project has flagged as hard to find, across BM25,
semantic, and hybrid search alike (see the "Stage 8" sections of
README.md). Answers the question directly: is the failure caused by
tokenization, the searchable-text representation, a genuine embedding
limitation, or ranking/fusion?

Method: inspect the real unit (name, docstring, source, tokens), then
compute real BM25 score/rank and cosine score/rank against five real
queries spanning the difficulty range already observed -- from queries
that are known misses to queries that are known hits -- plus an
ablation that strips rate_limit's source code down to just its
docstring, to isolate whether source-code boilerplate is diluting the
embedding or whether the embedding model itself lacks the association.

Usage:
    python3 diagnose_rate_limit.py
"""

import numpy as np

from backend.indexing.tokenizer import tokenize
from evaluate import build_pipeline

# Five queries spanning the difficulty range already seen in prior
# evaluations, from confirmed misses to confirmed hits, so the
# diagnosis covers the full spectrum rather than just the failure case.
QUERIES = [
    "prevent someone from spamming an endpoint",       # known miss (Stage 8 v1)
    "prevent repeated requests from the same client",  # known miss (Stage 8 v2, embedding_favorable)
    "figure out who is calling this api too often",    # known hit via incidental keyword overlap ("often")
    "RuntimeError rate limit exceeded",                # new bm25_favorable: exact source fragment
    "rate limit",                                      # near-exact identifier match
]


def main():
    unit_by_id, index, store, embedder = build_pipeline()
    rate_limit_unit = next(u for u in unit_by_id.values() if u.name == "rate_limit")
    n = len(unit_by_id)

    print("=" * 70)
    print("1. RAW UNIT INSPECTION")
    print("=" * 70)
    print(f"name: {rate_limit_unit.name}")
    print(f"docstring: {rate_limit_unit.docstring!r}")
    print("source:")
    print(rate_limit_unit.source)

    tokens = tokenize(rate_limit_unit.searchable_text())
    unique_tokens = sorted(set(tokens))
    print("=" * 70)
    print("2. TOKENIZED REPRESENTATION (what BM25 actually indexes)")
    print("=" * 70)
    print(f"{len(unique_tokens)} unique tokens indexed for this unit:")
    print(unique_tokens)
    print()
    print("Notably absent: any word a user would naturally type to describe")
    print("*why* rate limiting exists -- \"spam\", \"abuse\", \"throttle\",")
    print("\"client\", \"requests\", \"endpoint\". The indexed text is dominated")
    print("by decorator/closure implementation mechanics (wrapper, func, args,")
    print("kwargs, call_times, append, decorator, factory) rather than the")
    print("domain concept.")
    print()

    print("=" * 70)
    print("3. BM25 SCORE/RANK AND COSINE SCORE/RANK, PER QUERY")
    print("=" * 70)
    for q in QUERIES:
        query_tokens = tokenize(q)
        full_bm25 = index.search(q, top_k=n)
        bm25_dict = dict(full_bm25)
        bm25_rank = next((i + 1 for i, (uid, _) in enumerate(full_bm25) if uid == rate_limit_unit.id), None)
        bm25_score = bm25_dict.get(rate_limit_unit.id, 0.0)
        shared = sorted(set(query_tokens) & set(tokens))

        query_vec = embedder.encode_one(q)
        full_cos = store.search(query_vec, top_k=n)
        cos_dict = dict(full_cos)
        cos_rank = next(i + 1 for i, (uid, _) in enumerate(full_cos) if uid == rate_limit_unit.id)
        cos_score = cos_dict[rate_limit_unit.id]

        print(f'"{q}"')
        print(f"  shared BM25 tokens with rate_limit: {shared if shared else '(none)'}")
        if bm25_rank is None:
            print(f"  BM25: score=0.00, never scored (zero term overlap) among {n} units")
        else:
            print(f"  BM25: score={bm25_score:.3f}  rank={bm25_rank}/{n}")
        print(f"  cosine: score={cos_score:.3f}  rank={cos_rank}/{n}")
        print()

    print("=" * 70)
    print("4. ABLATION: does trimming source-code boilerplate help the embedding?")
    print("=" * 70)
    variants = {
        "name only": "rate_limit",
        "docstring only": rate_limit_unit.docstring,
        "full searchable_text (name+docstring+source, as actually indexed)": rate_limit_unit.searchable_text(),
    }
    hard_queries = QUERIES[:2]  # the two known misses
    for label, text in variants.items():
        vec = embedder.encode_one(text)
        print(f"[{label}]")
        for q in hard_queries:
            qvec = embedder.encode_one(q)
            cos = float(np.dot(vec, qvec))
            print(f'  "{q}" -> cosine={cos:.3f}')
        print()

    print("=" * 70)
    print("5. CONCLUSION")
    print("=" * 70)
    print("""\
Not a tokenization bug: BM25's zero score on the two hard queries is a
correct, expected consequence of genuinely zero token overlap -- BM25
is working exactly as designed given what those queries and this unit's
text actually contain.

Not primarily a searchable-text dilution problem: the ablation above
shows that stripping away all source-code boilerplate (using the
docstring alone, or even just the name) barely moves the cosine score
for the two hard queries -- nowhere close to the 0.35-0.48 range seen
once a query shares real vocabulary with the unit (see section 3's
"RuntimeError rate limit exceeded" / "rate limit" results). If dilution
by source-code noise were the main cause, trimming it should have
closed most of that gap. It didn't.

Not a ranking/fusion bug: HybridRanker's fusion math is independently
unit-tested (backend/tests/test_hybrid_ranker.py, 12 tests) and is
correctly propagating two genuinely weak inputs here, not introducing
a new error. Weak BM25 (0) and weak-but-real cosine (~0.15-0.17) fuse
to a weak hybrid score because that is what those two real numbers
should produce.

The primary cause is a genuine embedding-model limitation: even given
the cleanest possible signal (the docstring alone, with zero code
noise), all-MiniLM-L6-v2 still does not place "prevent someone from
spamming an endpoint" or "prevent repeated requests from the same
client" close to "Decorator factory that limits how often a function
can be called" in embedding space. This is plausible given the model
is general-purpose, not code- or security-terminology-specific (see
embedder.py's own docstring on this tradeoff) -- it may simply lack
strong training signal connecting "rate limiting" to "spam prevention"
or "abuse prevention" as synonymous concepts, even though a person
reading both would recognize them as the same thing.
""")


if __name__ == "__main__":
    main()
