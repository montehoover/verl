"""
Utilities for parsing user-provided text input.

This module provides a single function, parse_user_input, which tokenizes
a string into words suitable for text analysis.
"""

import re


# Pre-compiled regular expression pattern for improved performance and clarity.
# This pattern matches words composed of Unicode letters and digits, excluding
# underscores, and allows for internal apostrophes (e.g., don't, it's).
WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)


def parse_user_input(text):
    """
    Split a user-provided string into a list of words.

    The tokenizer is Unicode-aware, excludes underscores, and preserves
    internal apostrophes within words.

    Parameters:
        text (str): The input string provided by the user.

    Returns:
        list: A list of words extracted from the input string.

    Raises:
        ValueError: If the input is not a string or if an unexpected error
            occurs during processing.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    try:
        words = WORD_PATTERN.findall(text)
        return words
    except Exception as exc:
        # Normalize any unexpected error into ValueError to provide a
        # consistent interface to callers.
        raise ValueError(f"Failed to parse user input: {exc}") from exc
