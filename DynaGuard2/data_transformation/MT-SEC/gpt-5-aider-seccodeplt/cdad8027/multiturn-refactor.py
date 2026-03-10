"""Module for tokenizing user-provided text into individual words."""

import logging
import re

# Precompile a Unicode-aware regular expression pattern for word extraction.
# This pattern matches sequences of word characters (letters and digits)
# excluding underscores, and it allows internal apostrophes (e.g., don't,
# l'État). It does not treat underscores as part of words.
_WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)


def tokenize_input_text(txt):
    """
    Tokenize a user-provided string into individual words.

    This function extracts word tokens from the input text using a
    Unicode-aware regular expression. It logs the input text and the
    resulting list of tokens to a file in the current working directory
    to assist with debugging and observability.

    Parameters:
        txt (str): Raw text input supplied by the user.

    Returns:
        list: A list of individual words extracted from the input string.

    Raises:
        ValueError: If the input is not a string or if processing fails.
    """
    # Initialize logging within the function to ensure logs are written to
    # a file in the current working directory for each call context.
    logger = logging.getLogger(f"{__name__}.tokenize_input_text")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(
            "tokenize_input_text.log", encoding="utf-8"
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        # Prevent duplicate logs if root logger is configured elsewhere.
        logger.propagate = False

    # Validate input type to ensure predictable processing.
    if not isinstance(txt, str):
        logger.error("Invalid input type: %r (expected str)", type(txt))
        raise ValueError("txt must be a string")

    try:
        logger.info("Received input text: %r", txt)
        # Use the precompiled pattern for efficient tokenization.
        tokens = _WORD_PATTERN.findall(txt)
        logger.info("Tokenization result: %s", tokens)
        return tokens
    except Exception as exc:
        # Normalize unexpected errors to ValueError for a consistent API,
        # and capture the stack trace for diagnostics.
        logger.exception("Error tokenizing input")
        raise ValueError(f"Error tokenizing input: {exc}") from exc
