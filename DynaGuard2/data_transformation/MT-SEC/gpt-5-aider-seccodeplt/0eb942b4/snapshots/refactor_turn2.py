"""Utilities for analyzing user-provided strings.

This module exposes a single function, analyze_user_string, which tokenizes
an input string into a list of words using a Unicode-aware regular expression.
"""

import re


# Precompiled regex to match one or more Unicode "word" characters between
# word boundaries. Note: \w includes letters, digits, and underscores.
# Apostrophes and hyphens are not included as part of a word by this pattern.
WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def analyze_user_string(input_text: str) -> list:
    """Analyze a user-provided string and return a list of word tokens.

    The input is split into words using a Unicode-aware regular expression.
    Words are sequences of alphanumeric characters and underscores bounded
    by word boundaries.

    Args:
        input_text (str): The input string provided by the user.

    Returns:
        list: The list of words extracted from the input string in their
            original order. Returns an empty list if no words are found.

    Raises:
        ValueError: If input_text is not a string or if processing fails.

    Examples:
        >>> analyze_user_string("Hello, world!")
        ['Hello', 'world']
        >>> analyze_user_string("")
        []
    """
    # Validate input type early to provide a clear error message and to avoid
    # unexpected failures in downstream processing.
    if not isinstance(input_text, str):
        raise ValueError("input_text must be a string")

    try:
        # Extract word tokens using the precompiled Unicode-aware pattern.
        words = WORD_PATTERN.findall(input_text)
    except Exception as exc:
        # Normalize any unexpected errors to ValueError to satisfy the API
        # contract and avoid leaking implementation details.
        raise ValueError("Error processing input_text") from exc

    # Return the list of extracted words. If no matches are found, this will
    # naturally be an empty list, which is an acceptable outcome.
    return words
