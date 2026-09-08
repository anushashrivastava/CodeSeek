# CodeSeek

### AI-Powered Semantic & Hybrid Code Search Engine

A search engine for Python codebases: search using natural language
and get back relevant functions/classes even when your query doesn't
share exact words with the code. Combines traditional keyword search
(BM25) with AI-based semantic search (sentence embeddings), merged via
a configurable hybrid ranker — with a second fusion strategy
(Reciprocal Rank Fusion) implemented and evaluated alongside it.

**Target use:** portfolio project for Software Engineering interviews.
Emphasis is on understanding every component well enough to explain
it — including the parts that didn't work as expected — not on
maximizing features.

If you're continuing this project with Claude (claude.ai, Claude Code,
or another session), **paste this entire README as your first
message**, with: "Continue building this project from where it left
off." Everything needed to pick up seamlessly is below.

---

## Quick start

```bash
# 1. Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the test suite (offline, no network needed)
python3 -m pytest backend/tests/ -v

# 3. Verify real semantic + hybrid search against sample_repo/
#    (downloads ~80MB model on first run)
python3 verify_embeddings.py
python3 verify_hybrid_ranker.py

# 4. Run the API, then explore /docs in a browser
uvicorn backend.api.main:app --reload

# 5. Serve the frontend (in a second terminal) and use the UI
cd frontend && python3 -m http.server 5500
# then open http://127.0.0.1:5500/index.html

# 6. Run the evaluation suite (methodology + all real results below)
python3 evaluate.py          # original + expanded benchmarks + method comparison
python3 tune_alpha.py        # alpha sweep for the min-max hybrid ranker
python3 diagnose_rate_limit.py   # why one specific unit is hard to find
```

`curl` examples once the API is running:
```bash
curl -X POST http://127.0.0.1:8000/index -H "Content-Type: application/json" -d "{\"repo_path\": \"sample_repo\"}"
curl "http://127.0.0.1:8000/search?q=verify+someone+is+who+they+claim+to+be"
```

---

## Project goal and constraints

Point at a Python repository and search it in natural language — e.g.
"Where are user credentials validated?" — and get back relevant
functions/classes even if the exact words don't appear in the code.
Three pieces, as specified from the start:

1. Traditional keyword-based code search (BM25 inverted index)
2. AI-based semantic search (sentence embeddings + cosine similarity)
3. A ranking layer that combines both

Constraints that shaped every decision below:
- No LangChain/LlamaIndex, no agents, no RAG framework
- No Docker/Kubernetes/Postgres/Redis/vector DB unless truly justified
- Every important concept understood well enough to explain in a
  technical interview — not just working code
- Never fabricate results, benchmark numbers, or claims of things
  being tested when they weren't

---

## Architecture

```
Python Repository
       |
       v
Repository Scanner   -- finds safe .py files, ignores junk/secrets
       |
       v
AST Parser           -- extracts functions/classes/methods via ast module
       |
       v
CodeUnit objects      -- {file_path, name, type, lines, source, docstring}
      /        \
     v          v
Inverted Index   Embeddings
(BM25 keyword)   (semantic, cosine similarity)
      \          /
       v        v
  Ranking layer         -- HybridRanker (min-max, default) or
       |                   RRFRanker (rank fusion, evaluated alternative)
       v
     FastAPI            -- /health, /index, /search
       |
       v
  Simple Web UI          -- frontend/index.html, plain HTML/JS
```

Two independent retrieval systems (keyword, semantic) feed one ranking
layer. Neither depends on the other, so each is evaluated in isolation
before measuring any combination — and the ranking layer itself is
swappable: two fusion strategies exist behind the same interface, and
the evaluation below measures both honestly rather than assuming
either is better.

---

## Components

### Parsing (`backend/parser/`)
- **`models.py` — `CodeUnit`**: the one data structure everything
  downstream consumes — a function, class, or method with its file
  path, name, type, line range, source text, and docstring.
  `searchable_text()` concatenates name + docstring + source; this
  concatenation choice affects both keyword and semantic search
  quality, and turned out to matter a lot in practice (see the
  `rate_limit` diagnosis under Limitations).
- **`scanner.py`**: walks a repo directory, returns safe `.py` file
  paths. Ignores `.git`, `__pycache__`, `node_modules`, `venv`, `.env`,
  oversized files. Path-traversal-safe (resolves and validates paths
  stay within the repo root) — a repository is treated as untrusted
  input throughout this project.
- **`ast_parser.py`**: parses source via Python's built-in `ast`
  module (never executes code). Uses recursive descent with explicit
  scope tracking (an `in_function` flag) so nested/closure functions
  (e.g. a `wrapper` defined inside a `decorator` defined inside
  `rate_limit`) are walked but never indexed as separate units — a
  flat `ast.walk()` would double-count them as near-duplicate results.
  Malformed files are caught and skipped, not crashed on.

### Keyword search — BM25 (`backend/indexing/`)
- **`tokenizer.py`**: splits identifiers on snake_case/camelCase
  boundaries (so "credentials" matches `check_credentials`), keeps
  both the split pieces and the original compound token, lowercases,
  and filters a small code-specific stopword list (`self`, `def`,
  `return`, ...) rather than a generic English list, since code's own
  high-frequency low-signal tokens differ from English's. **No
  stemming** — this is a real, hand-verified limitation (see
  Limitations).
- **`inverted_index.py`**: `term -> {unit_id: term_frequency}`
  postings, ranked with hand-rolled BM25 (`k1=1.5`, `b=0.75`, standard
  Lucene/Elasticsearch defaults) — not a library like `rank_bm25`,
  specifically to understand the ranking math well enough to explain
  it. BM25 over raw TF-IDF for term-frequency saturation (`k1`) and
  document-length normalization (`b`, so long documents don't win
  purely by being long) — both verified by dedicated tests.

### Semantic search — embeddings (`backend/embeddings/`)
- **`embedder.py`**: wraps `SentenceTransformer('all-MiniLM-L6-v2')`,
  a small (~80MB), general-purpose, **not code-specific** model — a
  deliberate tradeoff for this project's scope, and the leading
  hypothesis behind the one real search gap found (see Limitations).
  `normalize_embeddings=True` so every vector has unit length, meaning
  a plain dot product already equals cosine similarity.
- **`vector_store.py`**: brute-force cosine similarity via one
  matrix-vector product, not FAISS or an ANN index. O(N) per query is
  fast enough at this scale (hundreds–low thousands of units); an ANN
  index earns its complexity at a scale this project doesn't have.

### Ranking layer (`backend/retrieval/`)

Two independent fusion strategies exist behind the same shape
(`(inverted_index, vector_store, embedder)` constructor,
`.search(query, top_k)` method), evaluated side by side rather than
one replacing the other:

- **`hybrid_ranker.py` — `HybridRanker` (the default).** Runs both
  underlying searches exhaustively (`top_k` = full corpus size, not
  just each method's own top-10) so every unit has a real score from
  both methods, then rescales each method's raw scores to [0, 1] via
  per-query min-max normalization before combining:
  `hybrid_score = alpha * cosine_normalized + (1 - alpha) * bm25_normalized`.
  Normalization is mandatory, not a nicety: real measured cosine scores
  sit in a narrow ~0.15–0.48 band while BM25 is unbounded and much
  larger — averaging raw scores would let BM25 dominate regardless of
  relevance. `alpha` defaults to 0.5 and is a constructor argument, not
  hardcoded, since the right balance is an empirical question (see
  Evaluation below — it's still 0.5; see why in Limitations).
  `HybridResult` exposes both raw and normalized component scores, not
  just the fused number, so any ranking is explainable ("won on
  keywords, not semantics").
- **`rrf_ranker.py` — `RRFRanker` (an evaluated alternative).** Fuses
  *rank positions* instead of scores: for each unit, sum
  `1 / (k + rank)` over every ranked list it appears in (`k=60`, the
  standard default from the original Cormack/Clarke/Buettcher paper).
  Never needs to normalize two differently-scaled score distributions
  against each other — appealing in principle, but not a strict
  improvement in practice (see Evaluation).

Both take the embedder via a `QueryEmbedder` Protocol (structural
typing) rather than a concrete import, so tests inject a fake embedder
and stay offline with no real model needed.

### API (`backend/api/main.py`)
Thin FastAPI wrapper with no search/ranking/indexing logic of its
own — `POST /index` calls the shared `build_search_pipeline()` (see
below), `GET /search` calls `HybridRanker`. `GET /health` is liveness
only. A single global in-memory `IndexState` holds "the currently
indexed repository" — no database, since this project has no
multi-repo/multi-user requirement; re-indexing replaces it wholesale.
`/search` before any `/index` call returns `409` (a client error, not
an empty result); `top_k`/`alpha` are range-validated via FastAPI's own
`Query(ge=..., le=...)`. The embedder is injected via `Depends`, so
tests override it with a fake, keeping `test_api.py` offline too. CORS
is wide open (`allow_origins=["*"]`) — a deliberate, scoped choice for
a local single-user dev tool, not a production security boundary.

### Shared pipeline (`backend/pipeline.py`)
`build_search_pipeline(repo_path, embedder)`: scan → parse → build
`InvertedIndex` → embed → build `VectorStore`, returning
`(all_units, index, store)`. This exact sequence used to be duplicated
independently across the API, `evaluate.py`, and both `verify_*.py`
scripts; it's now defined once here and imported everywhere, so a
future change to how indexing works only needs updating in one place.

### Frontend (`frontend/index.html`)
Single self-contained file — inline CSS + vanilla JS, no framework, no
build step, matching this project's "no build tooling" decision (there
is no state complexity here that would justify one). Has an indexing
panel, a search panel (with a `top_k` field and an `alpha` slider), and
a results list with per-result BM25/cosine score bars — the same
explainability thread running from `HybridResult` through the API into
the UI — plus click-to-expand source viewing. All dynamic text is
escaped before insertion into the DOM and code snippets are set via
`textContent`, never `innerHTML`, since a scanned repository is
untrusted input and a function/file name containing HTML/script-like
text must not be able to execute in the browser.

### Tests (`backend/tests/`) — 65 tests, all offline
- `test_ast_parser.py` (12) — scanner + parser correctness: junk-dir
  filtering, malformed files, nested-closure exclusion, async support.
- `test_inverted_index.py` (12) — tokenizer + BM25 correctness,
  including the document-length-penalty test.
- `test_vector_store.py` (8) — cosine similarity correctness against
  hand-built vectors with known expected similarities.
- `test_hybrid_ranker.py` (12) — min-max fusion math: alpha isolation,
  weighted blending, zero-overlap handling, normalizer edge cases.
- `test_rrf_ranker.py` (10) — RRF formula correctness against
  hand-calculated values, missing-rank handling, the "appearing in
  both lists beats a #1 rank in only one" mechanic, `k`'s damping
  effect.
- `test_api.py` (9) — endpoint behavior via `TestClient` + a fake
  embedder: health check, 409/400/422 error paths, full score
  breakdown present, re-indexing replaces the previous repo.
- `test_pipeline.py` (2) — dedicated end-to-end coverage for
  `build_search_pipeline()`: a real synthetic repository on disk
  through parsing, indexing, and a `HybridRanker` search in one chain
  (complementing `test_api.py`'s HTTP-layer coverage of the same
  chain), plus the empty-repository edge case.

### `sample_repo/`
A small synthetic Python repo used as ground truth throughout
development and for evaluation — 35 real code units across `auth.py`,
`database.py`, `users.py`, `routes.py`, `utils/security.py` — plus
deliberately "hostile" fixtures the scanner must ignore or skip:
`broken_syntax.py` (invalid Python), `.env` (secret file), `.git/`,
`venv/`, `node_modules/`.

---

## Evaluation

### Methodology

`evaluate.py` hand-labels real queries against `sample_repo/` with the
qualified name(s) of the code unit(s) that actually answer each one —
`"Class.method"` or the bare name, checked by hand against the real
source, never invented — then computes Precision@5 and Recall@5 for
every method being compared, all built from the **same**
`InvertedIndex`, `VectorStore`, and embedder, so only the
ranking/weighting differs between runs.

**Interpretation note, not a bug:** most queries have exactly one
relevant unit, so Precision@5 is structurally capped at `1/5 = 0.20`
even when the correct unit ranks 1st — the other 4 slots count as
misses by definition. **Recall@5 is the more informative metric** for
most of this label set.

The benchmark grew in three stages, kept in the code as clearly
separated, still-runnable pieces rather than silently overwritten:

1. **`PARAPHRASE_QUERIES` (30 queries).** Natural-language paraphrases
   of what each function does (e.g. "hash a user's password before
   storing it"). This was the original benchmark and its numbers are
   still reproduced exactly by `evaluate.py`'s first report.
2. **`CATEGORIZED_QUERIES` (24 more, 6 categories).** Added because the
   paraphrase style structurally favors semantic search — it couldn't
   tell whether BM25 had any role at all. Categories: `semantic`,
   `exact_identifier`, `partial_identifier`, `mixed`, `bm25_favorable`,
   `embedding_favorable`. Two of the six (`semantic`, `bm25_favorable`)
   didn't test what their names implied on the first attempt — checked
   by hand, `semantic`'s queries turned out to share real keyword
   overlap with their targets via docstring wording, so BM25 tied it
   trivially. Both were rebuilt and this time verified
   *programmatically*: every `semantic` query tokenized and confirmed
   to share **zero** tokens with its target's full indexed text; every
   `bm25_favorable` query rewritten as a literal source-fragment/jargon
   query (an exact exception message, an exact library call) and
   confirmed to share its tokens with the target. This history is kept
   visible in `evaluate.py`'s comments rather than quietly erased.
3. **Method comparison.** The same 54-query pool run through 4 methods:
   keyword-only, semantic-only, the min-max `HybridRanker`, and
   `RRFRanker` — answering not just "what's the best `alpha`" but "is
   min-max fusion, or fusion at all, actually the right approach."

A startup check in `build_pipeline()` raises loudly if any label
doesn't match a real parsed unit, so a typo in the ground truth can't
silently zero out a query's score and masquerade as a real finding.

### Results

**Mean Recall@5, final state, all four methods, by category** (the
headline table — see `evaluate.py`'s output for the full Precision@5
table and the `alpha`-sweep table across 0.0–1.0 in steps of 0.1):

| category | keyword-only | semantic-only | hybrid (α=0.5) | RRF (k=60) |
|---|---|---|---|---|
| **OVERALL** | 0.815 | **0.963** | 0.907 | 0.889 |
| paraphrase (original 30) | 0.900 | 0.967 | 0.933 | 0.933 |
| semantic | 0.000 | **1.000** | 0.500 | 0.500 |
| exact_identifier | 1.000 | 1.000 | 1.000 | 1.000 |
| partial_identifier | 0.750 | 1.000 | **1.000** | 0.750 |
| mixed | 1.000 | 1.000 | 1.000 | 1.000 |
| bm25_favorable | 1.000 | 1.000 | 1.000 | 1.000 |
| embedding_favorable | 0.500 | 0.750 | 0.750 | 0.750 |

**What this does and doesn't show:**

- **5 of 7 categories are saturated** at 1.000 recall for every method,
  including keyword-only in most of them. On this 35-unit corpus, most
  query styles here aren't hard enough to differentiate the methods.
- **The `semantic` category is the clearest positive result for
  semantic search anywhere in this project**: a verified-zero-overlap
  query set where BM25 gets exactly 0.000 recall and semantic-only
  gets 1.000 — a real capability gap BM25 structurally cannot close,
  demonstrated cleanly once the category was actually built to test
  that (see Methodology above).
- **`bm25_favorable`, even rebuilt twice with literal source-fragments
  and exact jargon, never found a query where BM25 was *necessary*** —
  semantic-only ties its perfect recall every time. That doesn't mean
  no such query exists on this corpus; it means none has been found
  yet (see Limitations).
- **Both fusion strategies tie-or-lose to semantic-only overall, via
  two different, root-caused failure modes:**
  - Min-max hybrid (`paraphrase` category, `alpha=0.5`): a query
    ("where do we check if someone is logged in") where semantic-only
    correctly finds `is_session_expired` but hybrid **misses it
    entirely** — several semantically-unrelated units with just enough
    incidental keyword overlap outweigh the true answer's real-but-
    modest cosine score once min-max normalized and averaged in.
  - RRF (`partial_identifier` category): the query "user validation" →
    `InputValidator.validate_username_format` has zero BM25 signal and
    an excellent cosine rank (2nd of 35), so under RRF it gets exactly
    one small contribution and ranks 19th overall — while five *less
    relevant* units each get two mediocre contributions (a weak BM25
    rank from incidental overlap on the word "user," plus a so-so
    cosine rank), and two mediocre contributions numerically outscore
    one excellent one. RRF's rank-only view discards the score
    *magnitude* that would tell you the difference between "barely
    present" and "genuinely absent" — min-max hybrid preserves that
    magnitude and still finds the answer here.
  
  These are mirror-image failure modes, not the same bug twice: one
  fusion strategy can be fooled by *inflated normalized scores*, the
  other by *structural bias toward consensus across two signals*.
  Neither is free of a failure mode.
- **`rate_limit` is the one unit every version of this benchmark has
  flagged as hard**, across every method, including a second
  independently-worded query in `embedding_favorable`
  ("prevent repeated requests from the same client") that fails at
  *every* `alpha` from 0.0 to 1.0. Diagnosed directly (`diagnose_rate_limit.py`,
  a permanent, reproducible script) — see Limitations for the
  conclusion.

### Does hybrid ranking — or RRF — actually earn its complexity?

Stated plainly, without favoring either implementation: **on this
project's benchmark, no interior fusion weight and no rank-based
fusion strategy was found to beat semantic-only overall.** An `alpha`
sweep (`tune_alpha.py`) confirmed no value between 0.0 and 1.0 beats
`alpha=1.0` (semantic-only) on the original 30-query set; the expanded
54-query set and the RRF comparison both reinforce the same direction.

This is **not** evidence that "semantic search is universally better"
or "keyword search / hybrid fusion is useless" — it's evidence that,
on this specific 35-unit synthetic corpus with the query styles tested
so far, semantic search alone was never beaten. Two structural reasons
this shouldn't be over-generalized are in Limitations below, and one
positive result for BM25 exists regardless of the aggregate numbers:
the `partial_identifier` category shows a real, root-caused case
(the tokenizer's lack of compound-word splitting) where BM25
structurally cannot find the answer and only added semantic weight
recovers it — the opposite failure from the ones described above, and
proof that keyword search's blind spots are real too, just not
dominant in this particular label set.

**`DEFAULT_ALPHA` remains `0.5`, unchanged.** `HybridRanker` remains
the default ranker in the codebase; `RRFRanker` is implemented, tested,
and evaluated as an alternative, not swapped in. Both decisions are
left open for the project owner rather than made unilaterally off one
benchmark — see Limitations for exactly why.

---

## Key engineering decisions

| Decision | Why |
|---|---|
| Python `ast` module, not regex/line-splitting | Exact function boundaries, same structure the interpreter itself uses |
| Recursive descent with `in_function` flag, not flat `ast.walk()` | Flat walk double-counted nested closures as separate units |
| Hand-rolled BM25, not a library (`rank_bm25`) | Understand the ranking math well enough to explain it, not hide it behind an import |
| Split identifiers on snake_case/camelCase | Otherwise "credentials" never matches `check_credentials` — the core motivating example from the spec |
| `normalize_embeddings=True` | Unit-length vectors mean dot product = cosine similarity |
| Brute-force NumPy cosine similarity, not FAISS | O(N) is fast enough at this scale; an ANN index solves a scale problem this project doesn't have |
| Local in-memory state, not Postgres/a vector DB | No multi-repo/multi-user requirement exists at this scale |
| Plain single-file HTML/JS frontend, not React | No state complexity here that would justify build tooling |
| Per-query min-max normalization before fusing BM25 + cosine | Measured real score ranges (cosine ~0.15-0.48, BM25 unbounded/larger) prove raw averaging would let BM25 dominate regardless of relevance |
| Exhaustive fusion (score every unit via both methods), not merging two separately-truncated top-k lists | Avoids silently dropping a unit one method ranked highly but the other placed just outside its own top-k |
| `alpha` and `k` as constructor arguments, not hardcoded | The right balance is an empirical question the evaluation script exists to answer |
| Both `HybridResult` and `RRFResult` expose raw/component data, not just the fused score | Every ranking is explainable — "won on keywords," "won on consensus across both signals" |
| `QueryEmbedder` Protocol instead of importing `CodeEmbedder` directly | Lets tests inject a fake embedder, keeping ranking tests offline |
| Embedder injected via FastAPI `Depends`, not a bare module import | Lets `test_api.py` override it with a fake, keeping API tests offline too |
| `/search` returns 409 (not 200 + empty list) before any `/index` call | An empty result there means "client forgot to index," a request error |
| `RRFRanker` added as a new, independent module, not a rewrite of `HybridRanker` | Evaluate an alternative without risking a working, tested implementation before knowing whether it's actually better |
| `build_search_pipeline()` extracted into `backend/pipeline.py` | The same scan→parse→index→embed sequence was duplicated across the API and 3 scripts; one implementation now, not four that could drift apart |
| Benchmark categories rebuilt via **programmatic** token-overlap verification, not by eye | Eyeballing is exactly what produced the original labeling mistake; a mechanical check against the real tokenizer can't repeat it |
| Kept `DEFAULT_ALPHA=0.5` and `HybridRanker` as the default despite evaluation results leaning toward semantic-only | The corpus is 35 units, two benchmark categories still haven't found a case for BM25, and changing a shipped default off one benchmark risks overfitting to that benchmark's query style — see Limitations |

---

## Limitations

Stated directly, not buried in caveats scattered across old build logs:

1. **The evaluation corpus is 35 code units.** Top-5 recall being
   saturated for most benchmark categories may simply reflect that a
   5-slot window rarely gets crowded out when there are only 35
   candidates total. A larger, real-world repository with many more
   near-duplicate or similarly-named functions could make BM25's
   exact-match precision matter in a way this corpus structurally
   cannot demonstrate. **The "semantic-only wins" finding above should
   not be assumed to generalize to larger codebases.**
2. **The embedding model is general-purpose, not code- or
   security-terminology-specific.** Direct diagnosis of the one
   consistent search failure (`rate_limit`, via
   `diagnose_rate_limit.py`) found: (a) BM25's zero score there is
   correct, not a bug — verified zero token overlap; (b) an ablation
   stripping all source-code boilerplate down to just the docstring
   barely moved the cosine score (0.146→0.147, 0.172→0.227) — nowhere
   near the 0.35–0.48 seen once a query shares real vocabulary with
   the unit — ruling out "too much code noise" as the main cause;
   (c) `HybridRanker`'s fusion math is independently unit-tested and
   correctly combining two genuinely weak real signals, not
   introducing a new error. The remaining, most plausible explanation:
   `all-MiniLM-L6-v2` may simply lack strong training signal connecting
   "rate limiting" to "spam/abuse prevention" as the same concept, even
   though a person reading both immediately would. A different query
   phrasing sharing even one incidental word ("often") *does* get
   found — showing keyword matching's occasional success here is
   fragile to word choice, not proof of real understanding.
3. **The tokenizer has no stemming and doesn't split plain lowercase
   compound words.** Confirmed directly: "username" is never split
   into "user" + "name" (only snake_case/camelCase boundaries are
   split), so a query like "user validation" shares zero BM25 tokens
   with `validate_username_format` even though the words are clearly
   related. This is the one clean case in this evaluation where BM25's
   blind spot is structural and semantic search's addition is
   unambiguously necessary — and also the case where RRF specifically
   underperforms min-max hybrid (see Evaluation results above).
4. **Two of six categorized-benchmark categories were rebuilt once
   already** (`semantic`, `bm25_favorable`) after failing to test what
   they claimed to. They're now verified programmatically, but a
   still-tighter or larger benchmark might yet surface a case for BM25
   that this one hasn't found. Treat "no case found" as "not found
   yet," not "proven impossible."
5. **No incremental indexing.** Re-indexing rebuilds everything from
   scratch and replaces the previous state wholesale; there's no
   multi-repository or multi-user support, by design, at this project's
   scale — see the `IndexState` decision above.
6. **`alpha` and RRF's `k` are both un-tuned defaults**, not results of
   a search for an optimal value. The evaluation deliberately avoided
   tuning either after seeing an unfavorable result, to prevent
   flattering a specific benchmark run — see the design-decisions
   table. Both remain open, explicit decisions for the project owner.
7. **The frontend and API have no authentication, rate limiting, or
   production hardening** — appropriate for a local single-user dev
   tool (the explicit scope), not for exposing this to untrusted
   users or networks.

---

## Status

| # | Stage | Status |
|---|---|---|
| 1 | AST parsing (scanner + parser + `CodeUnit`) | ✅ Done, tested |
| 2 | Keyword search (tokenizer + inverted index + BM25) | ✅ Done, tested |
| 3 | Embeddings (embedder + vector store) | ✅ Done, tested, verified with the real model |
| 4 | Semantic search integration | ✅ Done — wired into the ranking layer |
| 5 | Hybrid ranking (min-max) | ✅ Done, tested, verified with real fused rankings |
| 6 | FastAPI (`/index`, `/search`, `/health`) | ✅ Done, tested, verified live |
| 7 | Frontend (plain HTML/JS) | ✅ Done, verified in a real browser end-to-end |
| 8 | Evaluation (Precision@5/Recall@5, incl. RRF) | ✅ Done — see Evaluation above for full real results |
| 9 | Optional: "Explain this code" LLM button | ⬜ Not started, optional |
| 10 | Final polish | ✅ This pass: codebase deduplicated (`backend/pipeline.py` extracted), dead code removed, end-to-end tests added, README restructured |

**Test suite: 65/65 passing.** Verified live against `sample_repo/`
with the real model after this cleanup pass: `POST /index` →
`indexed_units: 35`; `GET /search?q=verify+someone+is+who+they+claim+to+be`
→ `check_credentials` first at `hybrid_score=1.000` — identical to
every prior verification, confirming the refactor changed no behavior.

Only Stage 9 (optional LLM feature) remains, and it's explicitly
optional — this project's core goal (keyword + semantic + hybrid
search, honestly evaluated) is complete.

---

## Project structure

```
code-search-engine/
├── backend/
│   ├── parser/
│   │   ├── models.py         # CodeUnit dataclass
│   │   ├── scanner.py        # safe file discovery
│   │   └── ast_parser.py     # AST -> CodeUnit extraction
│   ├── indexing/
│   │   ├── tokenizer.py      # code-aware tokenization
│   │   └── inverted_index.py # BM25 keyword search
│   ├── embeddings/
│   │   ├── embedder.py       # Sentence-Transformers wrapper
│   │   └── vector_store.py   # cosine similarity search
│   ├── retrieval/
│   │   ├── hybrid_ranker.py  # BM25 + cosine fusion via min-max normalization (default)
│   │   └── rrf_ranker.py     # BM25 + cosine fusion via Reciprocal Rank Fusion (evaluated alternative)
│   ├── api/
│   │   └── main.py           # FastAPI app: /health, /index, /search
│   ├── pipeline.py           # shared scan -> parse -> index -> embed pipeline
│   └── tests/
│       ├── test_ast_parser.py
│       ├── test_inverted_index.py
│       ├── test_vector_store.py
│       ├── test_hybrid_ranker.py
│       ├── test_rrf_ranker.py
│       ├── test_api.py
│       └── test_pipeline.py
├── frontend/
│   └── index.html            # plain HTML/JS/CSS UI, no build step
├── sample_repo/              # synthetic test repo (see Components above)
├── verify_embeddings.py      # validate real semantic search against sample_repo/
├── verify_hybrid_ranker.py   # validate real fused rankings against sample_repo/
├── evaluate.py                # full evaluation suite (see Evaluation above)
├── tune_alpha.py              # sweeps alpha 0.0-1.0 against evaluate.py's label set
├── diagnose_rate_limit.py     # why rate_limit is the hardest unit to find
├── requirements.txt
├── .gitignore
└── README.md                 # this file
```

---

## Engineering ground rules (still in force)

- No unnecessary abstractions, no dozens of trivial files, no
  microservices, no agents, no complex LLM orchestration.
- Treat repository input as untrusted: never execute repo code, guard
  against path traversal, ignore secrets, handle malformed files
  gracefully, limit oversized files.
- Every important concept explained (what/why/how/alternatives/
  interview angle) alongside implementation.
- Verify each stage via real execution before moving to the next.
- Never fabricate results, benchmark numbers, or claims of things
  being tested when they weren't — including when the result is
  unflattering to a component this project built.
