"""Walks a repository directory and returns the list of Python files
that are safe and worth parsing.

This module deliberately knows nothing about AST or code semantics.
Its only job is: given a directory, return a clean list of file paths.
Keeping this separate from the parser means we can test "did we find
the right files" independently from "did we parse them correctly."
"""

import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Directories we never want to walk into. These are either huge
# (virtual envs, node_modules), irrelevant (VCS internals, caches),
# or occasionally contain generated code that isn't representative
# of the actual project.
IGNORED_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",  # in case someone names a directory this
    "site-packages",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".tox",
}

# Files that must never be read, even if they'd otherwise match *.py,
# because they commonly contain secrets or are not real source files.
IGNORED_FILE_NAMES = {
    ".env",
}

MAX_FILE_SIZE_BYTES = 1_000_000  # 1 MB; skip anything larger as a safety limit


def scan_repository(repo_path: str) -> List[str]:
    """Return a list of absolute paths to Python files safe to parse.

    Security note: repo_path is treated as untrusted input. We resolve
    it to an absolute path and only ever walk *within* it, so a
    maliciously crafted relative path (e.g. containing "..") cannot
    cause us to read files outside the given repository root.
    """
    root = Path(repo_path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    python_files: List[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place so os.walk never descends
        # into them. This is far more efficient than walking everything
        # and filtering afterward, especially for huge venv/ folders.
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIR_NAMES]

        for filename in filenames:
            if filename in IGNORED_FILE_NAMES:
                continue
            if not filename.endswith(".py"):
                continue

            full_path = Path(dirpath) / filename

            # Path traversal guard: confirm the resolved file path is
            # still inside root. os.walk shouldn't escape root on its
            # own, but this makes the guarantee explicit and defensive.
            try:
                full_path.resolve().relative_to(root)
            except ValueError:
                logger.warning("Skipping file outside repo root: %s", full_path)
                continue

            try:
                if full_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    logger.warning("Skipping oversized file: %s", full_path)
                    continue
            except OSError as e:
                logger.warning("Could not stat file %s: %s", full_path, e)
                continue

            python_files.append(str(full_path))

    logger.info("Scanned %s: found %d Python files", root, len(python_files))
    return python_files


def to_relative_path(file_path: str, repo_path: str) -> str:
    """Convert an absolute file path back to a path relative to the repo root.

    Used so results/metadata show clean paths like 'utils/security.py'
    instead of the full absolute filesystem path.
    """
    return str(Path(file_path).resolve().relative_to(Path(repo_path).resolve()))
