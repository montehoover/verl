"""Module for tokenizing user-provided text into individual words."""

import re

# Precompile a Unicode-aware regular expression pattern for word extraction.
# This pattern matches sequences of word characters (letters and digits)
# excluding underscores, and it allows internal apostrophes (e.g., don't,
# l'État). It does not treat underscores as part of words.
_WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)


def tokenize_input_text(txt):
    """
    Tokenize a user-provided string into individual words.

    Parameters:
        txt (str): Raw text input supplied by the user.

    Returns:
        list: A list of individual words extracted from the input string.

    Raises:
        ValueError: If the input is not a string or if processing fails.
    """
    # Validate input type to ensure predictable processing.
    if not isinstance(txt, str):
        raise ValueError("txt must be a string")

    try:
        # Use the precompiled pattern for efficient tokenization.
        tokens = _WORD_PATTERN.findall(txt)
        return tokens
    except Exception as exc:
        # Normalize unexpected errors to ValueError for a consistent API.
        raise ValueError(f"Error tokenizing input: {exc}") from exc
