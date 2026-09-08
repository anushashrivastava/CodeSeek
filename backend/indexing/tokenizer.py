"""Tokenization for source code.

Code identifiers are not natural-language words, so naive whitespace
splitting misses most useful matches. This module splits identifiers
on snake_case and camelCase boundaries in addition to normal
punctuation/whitespace splitting, so a query term like "credentials"
can match a function named `check_credentials`.
"""

import re
from typing import List, Set

# Common Python/programming keywords and near-universal identifier
# fragments that appear so frequently they carry almost no
# discriminating signal for search (every function has "self", most
# have "return"). We filter these the same way a text search engine
# filters "the", "a", "is". This is a deliberately small, hand-picked
# list rather than a generic English stopword list, because code has
# its own distinct set of high-frequency low-information tokens.
CODE_STOPWORDS: Set[str] = {
    "self", "def", "class", "return", "if", "else", "elif", "for",
    "while", "import", "from", "as", "try", "except", "finally",
    "with", "pass", "none", "true", "false", "and", "or", "not",
    "in", "is", "lambda", "yield", "raise", "assert", "global",
    "nonlocal", "del", "async", "await",
}

# Matches a run of lowercase letters/digits followed by an uppercase
# letter, e.g. splits "checkCredentials" right before the "C".
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Matches any run of characters that are NOT letters/digits/underscore
# — used to split on punctuation, whitespace, operators, etc.
_NON_WORD = re.compile(r"[^a-zA-Z0-9_]+")


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Split source text (or a query) into normalized search tokens.

    Steps:
      1. Split on non-word characters (whitespace, punctuation, operators).
      2. Split each remaining chunk on snake_case ("_") boundaries.
      3. Split each piece further on camelCase boundaries.
      4. Lowercase everything.
      5. Keep both the split pieces AND the original compound identifier,
         so "check_credentials" yields "check", "credentials", AND
         "check_credentials" as searchable tokens.
      6. Optionally drop code stopwords and single-character tokens,
         which are almost always noise.
    """
    tokens: List[str] = []

    for raw_chunk in _NON_WORD.split(text):
        if not raw_chunk:
            continue

        # Fully split this chunk into its atomic pieces first (on both
        # snake_case and camelCase boundaries).
        split_pieces = []
        for snake_piece in raw_chunk.split("_"):
            if not snake_piece:
                continue
            for camel_piece in _CAMEL_CASE_BOUNDARY.split(snake_piece):
                piece = camel_piece.lower()
                if piece:
                    split_pieces.append(piece)

        tokens.extend(split_pieces)

        # Only add the original compound identifier as an *additional*
        # token when splitting actually produced something different
        # from a single plain word — otherwise "where" would be added
        # once as the compound and again as its own only split piece,
        # silently doubling its term frequency.
        compound = raw_chunk.lower()
        is_plain_word = len(split_pieces) == 1 and split_pieces[0] == compound
        if len(compound) > 1 and not is_plain_word:
            tokens.append(compound)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in CODE_STOPWORDS and len(t) > 1]

    return tokens
