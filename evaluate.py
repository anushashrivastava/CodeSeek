"""Stage 8: quantitative evaluation of keyword-only vs. semantic-only
vs. hybrid search, using a hand-labeled set of (query, relevant units)
pairs against sample_repo/.

This is the stage that turns anecdotal findings from earlier
verify_*.py runs (e.g. "rate_limit doesn't show up for this query")
into actual numbers, computed the same way for every method using the
same underlying index/store/embedder -- only `alpha` varies.

Metrics: Precision@5 and Recall@5, as specified in the project plan.

Important interpretation note, not a bug: many queries below have
exactly one relevant unit. For a single-relevant-item query,
Precision@5 is structurally capped at 1/5 = 0.20 even for a "perfect"
ranker that puts the right answer first -- Precision@k counts the
other k-1 slots as misses by definition. Recall@5 (did the relevant
unit appear anywhere in the top 5) is the more informative number for
most of these queries.

--- Benchmark history (read before changing EVAL_QUERIES) ---

v1 (PARAPHRASE_QUERIES, 30 queries): the original Stage 8 benchmark.
Mostly natural-language paraphrases of what a function does (e.g.
"hash a user's password before storing it"). Found that hybrid at
alpha=0.5 underperforms semantic-only (mean R@5 0.933 vs 0.967), and a
later alpha sweep (tune_alpha.py) found no interior alpha ever beats
semantic-only on this set. But this set is structurally biased toward
semantic search: paraphrasing docstrings is exactly the kind of task
sentence embeddings are trained for, so a benchmark built entirely
that way cannot fairly judge whether BM25 contributes anything.

v2 (this file, adds CATEGORIZED_QUERIES, 24 more queries): built to
test that bias directly, with categories deliberately chosen so some
should favor BM25 and some should favor embeddings -- see each
category's comment below. PARAPHRASE_QUERIES is kept completely
unchanged (same queries, same labels) so its numbers remain directly
comparable to what was already reported; it is now just one category
("paraphrase") among several, not a separate benchmark.

v3 (this file, fixes the `semantic` and `bm25_favorable` categories):
v2's own self-critique found both categories didn't test what their
names implied. `semantic`'s 4 queries all turned out to have real (if
indirect) keyword overlap via docstring wording, so BM25 tied it
trivially -- not evidence semantic queries are easy for BM25 in
general, just that those 4 labels weren't as lexically independent as
intended. `bm25_favorable` used natural-sentence jargon queries that
the embedding model also handled fine, so it never demonstrated BM25
being necessary either. Both were rebuilt and, this time, verified
*programmatically* against the real tokenizer before being accepted:
every `semantic` query is confirmed to share zero tokens with its
target's full indexed text (name+docstring+source); every
`bm25_favorable` query is confirmed to share its literal tokens with
its target (these are now source-fragment/jargon-style queries, e.g.
exact exception text or a library call, rather than natural sentences
that happen to mention a term). PARAPHRASE_QUERIES, and the other 4
CATEGORIZED_QUERIES categories (exact_identifier, partial_identifier,
mixed, embedding_favorable), are unchanged from v2.

v4 (this file, adds run_method_comparison()): adds Reciprocal Rank
Fusion (RRFRanker, backend/retrieval/rrf_ranker.py) as a 4th method
compared alongside keyword-only, semantic-only, and the existing
min-max HybridRanker -- on the exact same 54-query pool, overall and
per category. RRF fuses rank positions instead of scores, sidestepping
the score-normalization issues min-max fusion has (see hybrid_ranker.py
and rrf_ranker.py's module docstrings). HybridRanker itself is
unchanged; RRF is evaluated as an alternative, not a replacement.

Usage:
    python3 evaluate.py
"""

from backend.embeddings.embedder import CodeEmbedder
from backend.embeddings.vector_store import VectorStore
from backend.indexing.inverted_index import InvertedIndex
from backend.parser.models import CodeUnit
from backend.pipeline import build_search_pipeline
from backend.retrieval.hybrid_ranker import HybridRanker
from backend.retrieval.rrf_ranker import DEFAULT_RRF_K, RRFRanker

TOP_K = 5
ALPHA_GRID = [0.0, 0.3, 0.5, 0.7, 1.0]

# --- v1 benchmark: unchanged from the original Stage 8 evaluation. ---
# Each entry is a real query paired with the qualified name(s) of the
# code unit(s) that actually answer it, checked by hand against the
# real sample_repo/ source (see auth.py, database.py, users.py,
# routes.py, utils/security.py). Qualified name is "Class.method" for
# methods, or the bare name for functions/classes -- same convention
# used by the API and frontend.
#
# The last four repeat the four low-lexical-overlap queries used in
# verify_embeddings.py / verify_hybrid_ranker.py, so their anecdotal
# results (e.g. the rate_limit miss) get an actual Recall@5 number
# here instead of a one-off observation.
PARAPHRASE_QUERIES = [
    {"query": "hash a user's password before storing it", "relevant": ["hash_password"], "category": "paraphrase"},
    {"query": "verify a password against the stored hash", "relevant": ["check_credentials"], "category": "paraphrase"},
    {"query": "generate a session token for a logged-in user", "relevant": ["create_session_token"], "category": "paraphrase"},
    {"query": "check whether a session has timed out", "relevant": ["is_session_expired"], "category": "paraphrase"},
    {
        "query": "lock an account after too many failed login attempts",
        "relevant": ["LoginManager.record_failure", "LoginManager.is_locked_out"],
        "category": "paraphrase",
    },
    {"query": "clear the failed login counter after a successful login", "relevant": ["LoginManager.reset_attempts"], "category": "paraphrase"},
    {"query": "manages login attempts and account lockout policy", "relevant": ["LoginManager"], "category": "paraphrase"},
    {"query": "open a connection to the sqlite database file", "relevant": ["Database.connect"], "category": "paraphrase"},
    {"query": "close the active database connection", "relevant": ["Database.close"], "category": "paraphrase"},
    {"query": "run a parameterized sql query", "relevant": ["Database.execute"], "category": "paraphrase"},
    {"query": "commit the current database transaction", "relevant": ["Database.commit"], "category": "paraphrase"},
    {"query": "look up a user record by numeric id", "relevant": ["fetch_user_by_id"], "category": "paraphrase"},
    {"query": "look up a user record by their username", "relevant": ["fetch_user_by_username"], "category": "paraphrase"},
    {"query": "record an action taken by a user for auditing", "relevant": ["insert_audit_log"], "category": "paraphrase"},
    {"query": "create a brand new user account", "relevant": ["register_user"], "category": "paraphrase"},
    {"query": "deactivate a user's account without deleting their data", "relevant": ["deactivate_account"], "category": "paraphrase"},
    {"query": "update the email address on file for a user", "relevant": ["update_email"], "category": "paraphrase"},
    {"query": "serialize a user object to a plain dictionary", "relevant": ["User.to_dict"], "category": "paraphrase"},
    {"query": "get a human-friendly display name for a user", "relevant": ["User.display_name"], "category": "paraphrase"},
    {"query": "process an incoming login HTTP request", "relevant": ["handle_login_request"], "category": "paraphrase"},
    {"query": "process a new account signup request", "relevant": ["handle_signup_request"], "category": "paraphrase"},
    {"query": "invalidate a session token and log the user out", "relevant": ["handle_logout_request"], "category": "paraphrase"},
    {"query": "strip dangerous characters out of user-supplied text", "relevant": ["sanitize_input"], "category": "paraphrase"},
    {"query": "check whether a string is a syntactically valid email", "relevant": ["is_valid_email"], "category": "paraphrase"},
    {"query": "limit how often a function can be called in a time window", "relevant": ["rate_limit"], "category": "paraphrase"},
    {
        "query": "check that a username only contains allowed characters",
        "relevant": ["InputValidator.validate_username_format"],
        "category": "paraphrase",
    },
    {"query": "where do we check if someone is logged in", "relevant": ["is_session_expired"], "category": "paraphrase"},
    {"query": "verify someone is who they claim to be", "relevant": ["check_credentials"], "category": "paraphrase"},
    {"query": "prevent someone from spamming an endpoint", "relevant": ["rate_limit"], "category": "paraphrase"},
    {"query": "make sure user input is clean before using it", "relevant": ["sanitize_input"], "category": "paraphrase"},
]

# --- v2 benchmark: added to directly test whether hybrid search earns
# its complexity, by including query styles the v1 set had no examples
# of. Ground truth for every entry below was checked by hand against
# the real sample_repo/ source -- none of it is invented. Where a
# category's real BM25 behavior turned out to differ from the naive
# expectation (checked against backend/indexing/tokenizer.py, which
# has no stemming and does not split plain lowercase compound words
# like "username" into "user"+"name"), that is called out in a comment
# rather than adjusted to fit the expectation.
CATEGORIZED_QUERIES = [
    # Natural-language semantic queries: describe behavior while
    # sharing ZERO tokens with the target's full indexed text
    # (name+docstring+source). Verified programmatically (not just
    # eyeballed -- that's exactly how v2's "semantic" category ended
    # up accidentally overlapping) by tokenizing each query and each
    # target's searchable_text() and confirming an empty intersection.
    # Expected to favor embeddings, since BM25 has structurally zero
    # signal to work with here.
    {
        "query": "block someone from trying to sign in again once they've made too many mistakes",
        "relevant": ["LoginManager.is_locked_out"],
        "category": "semantic",
    },
    {"query": "confirm someone typed their electronic mail correctly", "relevant": ["is_valid_email"], "category": "semantic"},
    {
        "query": "turn off someone's profile without wiping out its information",
        "relevant": ["deactivate_account"],
        "category": "semantic",
    },
    {
        "query": "produce a unique string that proves someone is currently signed in",
        "relevant": ["create_session_token"],
        "category": "semantic",
    },

    # Exact identifier queries: the literal function/method name,
    # typed almost verbatim. Expected to strongly favor BM25 (and to
    # be trivial for it -- these are the easiest possible case).
    {"query": "insert_audit_log", "relevant": ["insert_audit_log"], "category": "exact_identifier"},
    {"query": "fetch_user_by_id", "relevant": ["fetch_user_by_id"], "category": "exact_identifier"},
    {"query": "check_credentials", "relevant": ["check_credentials"], "category": "exact_identifier"},
    {"query": "sanitize_input", "relevant": ["sanitize_input"], "category": "exact_identifier"},

    # Partial identifier queries: fragments of a real identifier, not
    # the whole thing and not a full sentence. Expected to favor BM25,
    # but only where the fragment actually matches a distinct token --
    # see the "user validation" note below.
    {"query": "audit log", "relevant": ["insert_audit_log"], "category": "partial_identifier"},
    # NOTE: "username" is one plain lowercase word in both the query's
    # target docstring and the identifier -- the tokenizer only splits
    # on snake_case/camelCase boundaries, so it is never split into
    # "user" + "name". This query's tokens ("user", "validation") are
    # therefore NOT expected to literally match anything in
    # InputValidator.validate_username_format's index entry. Kept
    # as-is deliberately: if BM25 scores this at 0, that is a real,
    # honest finding about the tokenizer's limits, not a mistake to
    # quietly fix by picking an easier query.
    {"query": "user validation", "relevant": ["InputValidator.validate_username_format"], "category": "partial_identifier"},
    {"query": "session token", "relevant": ["create_session_token"], "category": "partial_identifier"},
    {
        "query": "failed attempts",
        "relevant": ["LoginManager.record_failure", "LoginManager.is_locked_out"],
        "category": "partial_identifier",
    },

    # Mixed queries: a natural phrase built directly around real
    # identifier words. Expected to give both methods something to
    # work with.
    {"query": "validate username format", "relevant": ["InputValidator.validate_username_format"], "category": "mixed"},
    {"query": "hash password function", "relevant": ["hash_password"], "category": "mixed"},
    {"query": "fetch user by username", "relevant": ["fetch_user_by_username"], "category": "mixed"},
    {"query": "close database connection", "relevant": ["Database.close"], "category": "mixed"},

    # Code-terminology queries: literal source-fragment/jargon style
    # (an exact library call, an exact exception message), not a
    # natural sentence that merely mentions a related word -- v2's
    # bm25_favorable queries were natural sentences ("database
    # connection", "session expired") that the embedding model handled
    # just as well, so they never demonstrated BM25 being necessary.
    # Verified programmatically: every token in each query below is
    # confirmed present in its target's full indexed text.
    {"query": "sha256 hexdigest", "relevant": ["hash_password"], "category": "bm25_favorable"},
    {"query": "RuntimeError rate limit exceeded", "relevant": ["rate_limit"], "category": "bm25_favorable"},
    {"query": "cursor execute params", "relevant": ["Database.execute"], "category": "bm25_favorable"},
    {"query": "re.sub dangerous characters", "relevant": ["sanitize_input"], "category": "bm25_favorable"},

    # Low-lexical-overlap queries: paraphrased specifically to avoid
    # the target's own words. Expected to favor embeddings; a repeat
    # target (rate_limit) is included deliberately since it is the
    # unit every prior benchmark has already found hardest to surface.
    {"query": "prevent repeated requests from the same client", "relevant": ["rate_limit"], "category": "embedding_favorable"},
    {"query": "turn a raw password into something safe to store", "relevant": ["hash_password"], "category": "embedding_favorable"},
    {"query": "get rid of characters that could break a database query", "relevant": ["sanitize_input"], "category": "embedding_favorable"},
    {"query": "figure out who is calling this api too often", "relevant": ["rate_limit"], "category": "embedding_favorable"},
]

# The full benchmark pool. tune_alpha.py imports this name, so a
# future run of that script will use the full expanded pool, not just
# the original 30 -- intentional, since the whole point of expanding
# this set was that the original was too narrow to trust for tuning
# decisions.
EVAL_QUERIES = PARAPHRASE_QUERIES + CATEGORIZED_QUERIES

CATEGORY_DISPLAY_ORDER = [
    "paraphrase",
    "semantic",
    "exact_identifier",
    "partial_identifier",
    "mixed",
    "bm25_favorable",
    "embedding_favorable",
]


def qualified_name(unit: CodeUnit) -> str:
    """Same "Class.method" / bare-name convention used by the API and
    frontend, so labels in EVAL_QUERIES can be written by hand against
    the real source rather than against internal unit ids."""
    return f"{unit.parent_class}.{unit.name}" if unit.parent_class else unit.name


def precision_at_k(ranked_names: list, relevant: set, k: int) -> float:
    """Fraction of the top k results that are relevant."""
    top_k = ranked_names[:k]
    hits = sum(1 for name in top_k if name in relevant)
    return hits / k


def recall_at_k(ranked_names: list, relevant: set, k: int) -> float:
    """Fraction of all relevant units that appear in the top k results."""
    top_k = ranked_names[:k]
    hits = sum(1 for name in top_k if name in relevant)
    return hits / len(relevant)


def build_pipeline():
    """Scan, parse, and index sample_repo/ once; validate every
    EVAL_QUERIES label against the real parsed units; return
    (unit_by_id, index, store, embedder) for evaluation scripts to
    build HybridRanker instances from.

    Factored out of main() so other scripts (e.g. tune_alpha.py) can
    reuse the exact same pipeline and label set instead of duplicating
    either -- duplicating EVAL_QUERIES across files would risk the two
    copies silently drifting apart.
    """
    embedder = CodeEmbedder()
    all_units, index, store = build_search_pipeline("sample_repo", embedder)

    print(f"Parsed {len(all_units)} code units from sample_repo.\n")

    unit_by_id = {u.id: u for u in all_units}
    all_qualified_names = {qualified_name(u) for u in all_units}

    # Guard against a typo'd label in EVAL_QUERIES silently zeroing out
    # that query's score (it would look like a real ranking failure,
    # not a data-entry mistake) -- fail loudly instead.
    for entry in EVAL_QUERIES:
        for name in entry["relevant"]:
            if name not in all_qualified_names:
                raise ValueError(
                    f'EVAL_QUERIES label "{name}" for query "{entry["query"]}" '
                    f"does not match any parsed unit. Typo in the label?"
                )

    return unit_by_id, index, store, embedder


def score_queries(ranker: HybridRanker, unit_by_id: dict, queries: list) -> tuple:
    """Mean Precision@K / Recall@K for `ranker` over `queries`."""
    total_p = 0.0
    total_r = 0.0
    for entry in queries:
        relevant = set(entry["relevant"])
        results = ranker.search(entry["query"], top_k=TOP_K)
        ranked_names = [qualified_name(unit_by_id[r.unit_id]) for r in results]
        total_p += precision_at_k(ranked_names, relevant, TOP_K)
        total_r += recall_at_k(ranked_names, relevant, TOP_K)

    n = len(queries)
    return total_p / n, total_r / n


def run_original_benchmark(unit_by_id: dict, index: InvertedIndex, store: VectorStore, embedder: CodeEmbedder) -> None:
    """Reproduces the original Stage 8 report byte-for-byte in
    structure (same 30 queries, same 3 methods, same per-query print
    format), so anyone comparing against previously-reported numbers
    sees the same thing -- preserved rather than folded silently into
    the expanded report below."""
    print("#" * 60)
    print("ORIGINAL BENCHMARK (v1): 30 paraphrase-style queries")
    print("#" * 60)
    print()

    methods = {
        "keyword-only (alpha=0.0)": HybridRanker(index, store, embedder, alpha=0.0),
        "semantic-only (alpha=1.0)": HybridRanker(index, store, embedder, alpha=1.0),
        "hybrid (alpha=0.5)": HybridRanker(index, store, embedder, alpha=0.5),
    }

    totals = {name: {"precision": 0.0, "recall": 0.0} for name in methods}

    for i, entry in enumerate(PARAPHRASE_QUERIES, start=1):
        query = entry["query"]
        relevant = set(entry["relevant"])

        print(f'[{i:02d}/{len(PARAPHRASE_QUERIES)}] "{query}"')
        for method_name, ranker in methods.items():
            results = ranker.search(query, top_k=TOP_K)
            ranked_names = [qualified_name(unit_by_id[r.unit_id]) for r in results]

            p = precision_at_k(ranked_names, relevant, TOP_K)
            r = recall_at_k(ranked_names, relevant, TOP_K)
            totals[method_name]["precision"] += p
            totals[method_name]["recall"] += r

            print(f"    {method_name:26s} P@{TOP_K}={p:.2f}  R@{TOP_K}={r:.2f}")
        print()

    n = len(PARAPHRASE_QUERIES)
    print("=" * 60)
    print(f"Mean Precision@{TOP_K} / Recall@{TOP_K} across {n} queries (original benchmark):")
    print("=" * 60)
    for method_name, sums in totals.items():
        mean_p = sums["precision"] / n
        mean_r = sums["recall"] / n
        print(f"  {method_name:26s} mean P@{TOP_K}={mean_p:.3f}  mean R@{TOP_K}={mean_r:.3f}")
    print()


def run_expanded_benchmark(unit_by_id: dict, index: InvertedIndex, store: VectorStore, embedder: CodeEmbedder) -> None:
    """Overall + per-category Precision@5/Recall@5 across the full
    expanded query pool (54 queries: the original 30 plus 24 new,
    categorized queries), at 5 alpha values. No per-query printout
    here (54 queries x 5 alphas would be 270 lines) -- the per-query
    detail for the original 30 is already in run_original_benchmark();
    this section exists to answer the aggregate/category question,
    not to re-dump every individual query."""
    print("#" * 60)
    print(f"EXPANDED BENCHMARK (v2): {len(EVAL_QUERIES)} queries across {len(CATEGORY_DISPLAY_ORDER)} categories")
    print("#" * 60)
    print()

    by_category = _group_by_category(EVAL_QUERIES)

    for cat in CATEGORY_DISPLAY_ORDER:
        print(f"  {cat}: {len(by_category[cat])} queries")
    print()

    # Compute once per (category, alpha) -- one ranker per alpha,
    # reused across every category's queries -- rather than
    # recomputing search results separately for the precision and
    # recall tables below.
    row_groups = ["OVERALL"] + CATEGORY_DISPLAY_ORDER
    results = {cat: {} for cat in row_groups}
    for alpha in ALPHA_GRID:
        ranker = HybridRanker(index, store, embedder, alpha=alpha)
        for cat in row_groups:
            queries = EVAL_QUERIES if cat == "OVERALL" else by_category[cat]
            results[cat][alpha] = score_queries(ranker, unit_by_id, queries)

    print("Mean Precision@5 by alpha:")
    print(f"{'category':<20}" + "".join(f"{a:>9.1f}" for a in ALPHA_GRID))
    for cat in row_groups:
        row = f"{cat:<20}" + "".join(f"{results[cat][alpha][0]:>9.3f}" for alpha in ALPHA_GRID)
        print(row)
    print()

    print("Mean Recall@5 by alpha:")
    print(f"{'category':<20}" + "".join(f"{a:>9.1f}" for a in ALPHA_GRID))
    for cat in row_groups:
        row = f"{cat:<20}" + "".join(f"{results[cat][alpha][1]:>9.3f}" for alpha in ALPHA_GRID)
        print(row)
    print()


def _group_by_category(queries: list) -> dict:
    by_category = {cat: [] for cat in CATEGORY_DISPLAY_ORDER}
    for entry in queries:
        by_category[entry["category"]].append(entry)
    return by_category


def run_method_comparison(unit_by_id: dict, index: InvertedIndex, store: VectorStore, embedder: CodeEmbedder) -> None:
    """Stage 9: compares the 4 headline methods -- keyword-only,
    semantic-only, the existing min-max HybridRanker, and Reciprocal
    Rank Fusion (RRFRanker) -- overall and per category, on the same
    54-query pool used everywhere else in this file.

    This answers a different question than run_expanded_benchmark's
    alpha-grid table above: that table only sweeps HybridRanker's own
    alpha. This one asks whether a *different fusion strategy*
    (rank-based instead of score-based) does better, not just whether
    a different weight does.
    """
    print("#" * 60)
    print("METHOD COMPARISON (Stage 9): keyword vs semantic vs min-max hybrid vs RRF")
    print("#" * 60)
    print()

    methods = {
        "keyword-only": HybridRanker(index, store, embedder, alpha=0.0),
        "semantic-only": HybridRanker(index, store, embedder, alpha=1.0),
        "hybrid (a=0.5)": HybridRanker(index, store, embedder, alpha=0.5),
        f"RRF (k={DEFAULT_RRF_K})": RRFRanker(index, store, embedder, k=DEFAULT_RRF_K),
    }

    by_category = _group_by_category(EVAL_QUERIES)
    row_groups = ["OVERALL"] + CATEGORY_DISPLAY_ORDER

    # Compute once per (category, method) rather than recomputing
    # search results separately for the precision and recall tables
    # below -- same reasoning as run_expanded_benchmark above.
    results = {cat: {} for cat in row_groups}
    for method_name, ranker in methods.items():
        for cat in row_groups:
            queries = EVAL_QUERIES if cat == "OVERALL" else by_category[cat]
            results[cat][method_name] = score_queries(ranker, unit_by_id, queries)

    method_names = list(methods.keys())
    col_width = max(len(m) for m in method_names) + 2

    print("Mean Precision@5:")
    print(f"{'category':<20}" + "".join(f"{m:>{col_width}s}" for m in method_names))
    for cat in row_groups:
        row = f"{cat:<20}" + "".join(f"{results[cat][m][0]:>{col_width}.3f}" for m in method_names)
        print(row)
    print()

    print("Mean Recall@5:")
    print(f"{'category':<20}" + "".join(f"{m:>{col_width}s}" for m in method_names))
    for cat in row_groups:
        row = f"{cat:<20}" + "".join(f"{results[cat][m][1]:>{col_width}.3f}" for m in method_names)
        print(row)


def main():
    unit_by_id, index, store, embedder = build_pipeline()
    run_original_benchmark(unit_by_id, index, store, embedder)
    run_expanded_benchmark(unit_by_id, index, store, embedder)
    run_method_comparison(unit_by_id, index, store, embedder)


if __name__ == "__main__":
    main()
