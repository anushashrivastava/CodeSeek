"""Unit tests for tokenization and BM25 inverted index ranking."""

from backend.indexing.inverted_index import InvertedIndex
from backend.indexing.tokenizer import tokenize
from backend.parser.models import CodeUnit


# ---------- Tokenizer tests ----------

def test_tokenize_splits_snake_case():
    tokens = tokenize("check_credentials")
    assert "check" in tokens
    assert "credentials" in tokens
    assert "check_credentials" in tokens


def test_tokenize_splits_camel_case():
    tokens = tokenize("handleLoginRequest")
    assert "handle" in tokens
    assert "login" in tokens
    assert "request" in tokens


def test_tokenize_does_not_duplicate_plain_words():
    tokens = tokenize("database")
    assert tokens.count("database") == 1


def test_tokenize_removes_stopwords_by_default():
    tokens = tokenize("def check(self): return True")
    assert "self" not in tokens
    assert "return" not in tokens
    assert "def" not in tokens


def test_tokenize_lowercases():
    tokens = tokenize("CheckCredentials")
    assert all(t == t.lower() for t in tokens)


# ---------- Inverted index / BM25 tests ----------

def _make_unit(unit_id: int, name: str, source: str, docstring: str = None) -> CodeUnit:
    return CodeUnit(
        id=unit_id,
        file_path=f"{name}.py",
        name=name,
        unit_type="function",
        start_line=1,
        end_line=5,
        source=source,
        docstring=docstring,
    )


def test_search_returns_unit_containing_query_term():
    units = [
        _make_unit(0, "check_credentials", "def check_credentials(): pass",
                   "Verify user credentials"),
        _make_unit(1, "connect_database", "def connect_database(): pass",
                   "Open a database connection"),
    ]
    index = InvertedIndex()
    index.build(units)

    results = index.search("credentials")
    result_ids = [uid for uid, _ in results]

    assert 0 in result_ids
    assert 1 not in result_ids


def test_search_ranks_higher_term_frequency_higher():
    units = [
        _make_unit(0, "a", "auth auth auth logic here"),
        _make_unit(1, "b", "auth logic here but only mentioned once"),
    ]
    index = InvertedIndex()
    index.build(units)

    results = index.search("auth")
    result_ids = [uid for uid, _ in results]

    # Unit 0 mentions "auth" three times, unit 1 once -> 0 should rank first,
    # though BM25's saturation means it won't be 3x the score.
    assert result_ids[0] == 0


def test_search_returns_empty_list_for_unmatched_query():
    units = [_make_unit(0, "foo", "def foo(): pass", "Does foo things")]
    index = InvertedIndex()
    index.build(units)

    results = index.search("xyzxyz_nonexistent_term")
    assert results == []


def test_search_on_empty_index_returns_empty_list():
    index = InvertedIndex()
    index.build([])

    results = index.search("anything")
    assert results == []


def test_longer_document_is_penalized_for_same_term_frequency():
    # Both units mention "auth" exactly once, but unit 1 has a lot of
    # other unrelated padding, making it a much "longer" document.
    # BM25's length normalization (the b parameter) should penalize
    # the term's relative importance in the longer document.
    padding = " ".join(f"word{i}" for i in range(200))
    units = [
        _make_unit(0, "short_unit", "auth logic here"),
        _make_unit(1, "long_unit", f"auth logic here {padding}"),
    ]
    index = InvertedIndex()
    index.build(units)

    scores = dict(index.search("auth", top_k=10))
    assert scores[0] > scores[1]


def test_get_unit_returns_original_code_unit():
    unit = _make_unit(0, "foo", "def foo(): pass")
    index = InvertedIndex()
    index.build([unit])

    retrieved = index.get_unit(0)
    assert retrieved.name == "foo"
    assert retrieved is unit


def test_index_len_matches_number_of_units():
    units = [_make_unit(i, f"fn{i}", "pass") for i in range(5)]
    index = InvertedIndex()
    index.build(units)

    assert len(index) == 5
