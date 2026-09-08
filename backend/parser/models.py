"""Core data structures shared across the parser, indexer, and retrieval layers."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CodeUnit:
    """A single indexable unit of code: a function, class, or method.

    This is the atomic object that flows through the entire pipeline.
    The scanner finds files; the AST parser turns those files into a
    list of these. Everything downstream (inverted index, embeddings,
    ranking, API responses) operates on CodeUnit objects, never on
    raw file text directly.
    """

    id: int
    file_path: str          # path relative to the repo root, e.g. "utils/security.py"
    name: str                # e.g. "check_credentials" or "LoginManager"
    unit_type: str           # "function" | "class" | "method"
    start_line: int
    end_line: int
    source: str               # exact source text of this unit, as written
    docstring: Optional[str] = None
    parent_class: Optional[str] = None   # set for methods, e.g. "LoginManager"

    def searchable_text(self) -> str:
        """Text representation used for keyword and semantic indexing.

        We combine the name, docstring, and source because each carries
        different signal: the name is often the strongest keyword match,
        the docstring carries human-written intent (great for semantic
        search), and the source carries the actual implementation detail.
        """
        parts = [self.name]
        if self.docstring:
            parts.append(self.docstring)
        parts.append(self.source)
        return "\n".join(parts)

    def snippet(self, max_lines: int = 8) -> str:
        """A short preview of the source code, for display in search results."""
        lines = self.source.splitlines()
        if len(lines) <= max_lines:
            return self.source
        return "\n".join(lines[:max_lines]) + "\n    ..."
