"""Utilities for analyzing user-provided strings.

This module exposes a single function, analyze_user_string, which tokenizes
an input string into a list of words using a Unicode-aware regular expression.
"""

import logging
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

    Logging:
        Logging is initialized within this function. It logs the received
        input string and the resulting list of extracted words in a clear
        and readable format.

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
    # Initialize a dedicated logger for this function on first use to avoid
    # affecting global logging configuration.
    logger = logging.getLogger(__name__ + ".analyze_user_string")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    # Validate input type early to provide a clear error message and to avoid
    # unexpected failures in downstream processing.
    if not isinstance(input_text, str):
        logger.error("Invalid input type: %r (expected str)", type(input_text))
        raise ValueError("input_text must be a string")

    # Log the incoming input text for traceability.
    logger.info("Input text: %r", input_text)

    try:
        # Extract word tokens using the precompiled Unicode-aware pattern.
        words = WORD_PATTERN.findall(input_text)
    except Exception as exc:
        # Log the exception with traceback and normalize to ValueError.
        logger.exception("Error processing input text")
        raise ValueError("Error processing input_text") from exc

    # Log the result in a clear, readable format before returning.
    logger.info("Extracted words (%d): %s", len(words), words)

    # Return the list of extracted words. If no matches are found, this will
    # naturally be an empty list, which is an acceptable outcome.
    return words
