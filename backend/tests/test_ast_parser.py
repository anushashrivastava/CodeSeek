"""Unit tests for the scanner and AST parser.

These tests don't touch the real sample_repo/ on disk — they use
small inline source strings via tmp_path fixtures, so they run fast
and each test is self-contained and easy to reason about in isolation.
"""

import os
import textwrap

import pytest

from backend.parser.ast_parser import parse_file
from backend.parser.scanner import scan_repository


def write_file(tmp_path, relative_name: str, content: str) -> str:
    """Helper: write `content` to tmp_path/relative_name and return the full path."""
    full_path = tmp_path / relative_name
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(textwrap.dedent(content))
    return str(full_path)


# ---------- Scanner tests ----------

def test_scanner_finds_python_files(tmp_path):
    write_file(tmp_path, "a.py", "x = 1")
    write_file(tmp_path, "sub/b.py", "y = 2")

    files = scan_repository(str(tmp_path))

    assert len(files) == 2
    assert any(f.endswith("a.py") for f in files)
    assert any(f.endswith(os.path.join("sub", "b.py")) for f in files)


def test_scanner_ignores_junk_directories(tmp_path):
    write_file(tmp_path, "real.py", "x = 1")
    write_file(tmp_path, "__pycache__/cached.py", "x = 1")
    write_file(tmp_path, ".git/config.py", "x = 1")
    write_file(tmp_path, "node_modules/lib.py", "x = 1")
    write_file(tmp_path, "venv/site.py", "x = 1")

    files = scan_repository(str(tmp_path))

    assert len(files) == 1
    assert files[0].endswith("real.py")


def test_scanner_ignores_env_files(tmp_path):
    write_file(tmp_path, "app.py", "x = 1")
    write_file(tmp_path, ".env", "SECRET=abc")

    files = scan_repository(str(tmp_path))

    assert len(files) == 1
    assert files[0].endswith("app.py")


def test_scanner_raises_on_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_repository(str(tmp_path / "does_not_exist"))


def test_scanner_handles_empty_repository(tmp_path):
    files = scan_repository(str(tmp_path))
    assert files == []


# ---------- AST parser tests ----------

def test_parses_simple_function(tmp_path):
    path = write_file(tmp_path, "mod.py", '''
        def greet(name):
            """Say hello to someone."""
            return f"Hello, {name}"
    ''')

    units = parse_file(path, "mod.py", next_id_start=0)

    assert len(units) == 1
    assert units[0].name == "greet"
    assert units[0].unit_type == "function"
    assert units[0].docstring == "Say hello to someone."


def test_parses_class_and_methods_with_parent_tag(tmp_path):
    path = write_file(tmp_path, "mod.py", '''
        class Greeter:
            """Greets people."""

            def __init__(self, name):
                self.name = name

            def greet(self):
                """Return a greeting."""
                return f"Hello, {self.name}"
    ''')

    units = parse_file(path, "mod.py", next_id_start=0)

    types = {u.name: u.unit_type for u in units}
    parents = {u.name: u.parent_class for u in units}

    assert types == {"Greeter": "class", "__init__": "method", "greet": "method"}
    assert parents["__init__"] == "Greeter"
    assert parents["greet"] == "Greeter"
    assert parents["Greeter"] is None


def test_nested_inner_functions_are_not_indexed(tmp_path):
    path = write_file(tmp_path, "mod.py", '''
        def outer():
            """An outer function with a nested closure."""
            def inner():
                return 1
            return inner()
    ''')

    units = parse_file(path, "mod.py", next_id_start=0)

    names = [u.name for u in units]
    assert names == ["outer"]
    assert "inner" not in names
    # The inner function's source should still be present inside outer's
    # captured source text, since outer's source is a plain line range.
    assert "def inner" in units[0].source


def test_malformed_file_is_skipped_not_raised(tmp_path):
    path = write_file(tmp_path, "broken.py", "def broken(:\n    pass")

    units = parse_file(path, "broken.py", next_id_start=0)

    assert units == []


def test_ids_are_assigned_sequentially_from_start(tmp_path):
    path = write_file(tmp_path, "mod.py", '''
        def a():
            pass

        def b():
            pass
    ''')

    units = parse_file(path, "mod.py", next_id_start=10)

    ids = [u.id for u in units]
    assert ids == [10, 11]


def test_unit_without_docstring_has_none(tmp_path):
    path = write_file(tmp_path, "mod.py", '''
        def undocumented():
            return 42
    ''')

    units = parse_file(path, "mod.py", next_id_start=0)

    assert units[0].docstring is None


def test_async_function_is_captured(tmp_path):
    path = write_file(tmp_path, "mod.py", '''
        async def fetch_data():
            """Fetch data asynchronously."""
            return await something()
    ''')

    units = parse_file(path, "mod.py", next_id_start=0)

    assert len(units) == 1
    assert units[0].name == "fetch_data"
    assert units[0].unit_type == "function"
