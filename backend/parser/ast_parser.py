"""Parses Python source files into CodeUnit objects using the built-in
`ast` module.

Design note: we never execute repository code. `ast.parse()` only
builds a syntax tree from text — it does not run the code, import
modules, or evaluate anything. This is what makes it safe to point
at an untrusted repository.
"""

import ast
import logging
from typing import List, Optional

from backend.parser.models import CodeUnit

logger = logging.getLogger(__name__)

# Function/class bodies longer than this are truncated when stored,
# so one pathological 10,000-line generated file can't blow up
# memory or embedding costs later.
MAX_UNIT_LINES = 300


def parse_file(file_path: str, relative_path: str, next_id_start: int) -> List[CodeUnit]:
    """Parse a single Python file into a list of CodeUnit objects.

    Returns an empty list (rather than raising) if the file cannot be
    read or contains invalid syntax — malformed files are expected
    input from an untrusted repository, not a program bug, so we log
    and skip rather than crash the whole indexing run.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as e:
        logger.warning("Could not read file %s: %s", file_path, e)
        return []

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        logger.warning("Skipping file with invalid syntax %s: %s", file_path, e)
        return []

    source_lines = source.splitlines()
    units: List[CodeUnit] = []

    # We use a mutable single-element list as a simple counter cell so
    # the recursive helper can increment a shared id counter without
    # needing to be a class or use a global.
    counter = {"next_id": next_id_start}

    _visit_body(
        nodes=tree.body,
        in_function=False,
        parent_class=None,
        relative_path=relative_path,
        source_lines=source_lines,
        units=units,
        counter=counter,
    )

    return units


def _visit_body(
    nodes: List[ast.AST],
    in_function: bool,
    parent_class: Optional[str],
    relative_path: str,
    source_lines: List[str],
    units: List[CodeUnit],
    counter: dict,
) -> None:
    """Recursively walk a list of AST statements, capturing top-level
    classes/functions and class methods as CodeUnits, while explicitly
    skipping anything nested inside a function body.

    The `in_function` flag is the key piece of state: once we descend
    into a function's body, we set it to True and never capture units
    from inside it, no matter how deeply nested further defs are. This
    mirrors how lexical scoping actually works — a name defined inside
    a function is private to that function's scope.
    """
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            if not in_function:
                units.append(
                    _build_unit(
                        node=node,
                        unit_type="class",
                        relative_path=relative_path,
                        source_lines=source_lines,
                        unit_id=counter["next_id"],
                        parent_class=None,
                    )
                )
                counter["next_id"] += 1

                # Recurse into the class body to find its methods.
                # in_function stays False here: methods are top-level
                # relative to the class, so they should be captured.
                _visit_body(
                    nodes=node.body,
                    in_function=False,
                    parent_class=node.name,
                    relative_path=relative_path,
                    source_lines=source_lines,
                    units=units,
                    counter=counter,
                )
            # If a class is defined inside a function, we deliberately
            # don't recurse into it — it's a scoping edge case rare
            # enough in real code that treating it as "not indexed"
            # is a reasonable, explicit simplification.

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not in_function:
                unit_type = "method" if parent_class else "function"
                units.append(
                    _build_unit(
                        node=node,
                        unit_type=unit_type,
                        relative_path=relative_path,
                        source_lines=source_lines,
                        unit_id=counter["next_id"],
                        parent_class=parent_class,
                    )
                )
                counter["next_id"] += 1

            # Recurse into the function body regardless, but with
            # in_function=True, so anything nested inside is walked
            # (in case future logic needs it) but never captured as
            # its own unit.
            _visit_body(
                nodes=node.body,
                in_function=True,
                parent_class=None,
                relative_path=relative_path,
                source_lines=source_lines,
                units=units,
                counter=counter,
            )


def _build_unit(
    node: ast.AST,
    unit_type: str,
    relative_path: str,
    source_lines: List[str],
    unit_id: int,
    parent_class: Optional[str],
) -> CodeUnit:
    """Build a CodeUnit from an AST node, extracting its exact source
    text by line range and its docstring via ast.get_docstring().
    """
    start_line = node.lineno
    end_line = getattr(node, "end_lineno", start_line)

    # end_lineno was added in Python 3.8+; source_lines is 0-indexed
    # while AST line numbers are 1-indexed, hence the slicing offset.
    truncated_end = min(end_line, start_line + MAX_UNIT_LINES)
    source_text = "\n".join(source_lines[start_line - 1:truncated_end])

    docstring = ast.get_docstring(node)

    return CodeUnit(
        id=unit_id,
        file_path=relative_path,
        name=node.name,
        unit_type=unit_type,
        start_line=start_line,
        end_line=end_line,
        source=source_text,
        docstring=docstring,
        parent_class=parent_class,
    )
