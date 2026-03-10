"""
Utilities for parsing user-provided text input.

This module provides a single function, parse_user_input, which tokenizes
a string into words suitable for text analysis.
"""

import logging
import re


# Pre-compiled regular expression pattern for improved performance and clarity.
# This pattern matches words composed of Unicode letters and digits, excluding
# underscores, and allows for internal apostrophes (e.g., don't, it's).
WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)


def parse_user_input(text):
    """
    Split a user-provided string into a list of words.

    The tokenizer is Unicode-aware, excludes underscores, and preserves
    internal apostrophes within words. This function also initializes and
    emits logging statements to help trace processing steps and diagnose
    issues.

    Parameters:
        text (str): The input string provided by the user.

    Returns:
        list: A list of words extracted from the input string.

    Raises:
        ValueError: If the input is not a string or if an unexpected error
            occurs during processing.
    """
    # Initialize logging within the function to ensure logs are captured
    # even when this module is used standalone. Only configure basic
    # logging if no handlers are already present on the root logger.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )

    logger = logging.getLogger(__name__)
    logger.debug("parse_user_input called with text: %r", text)

    if not isinstance(text, str):
        logger.error("Invalid input type: %s (expected str)", type(text).__name__)
        raise ValueError("text must be a string")

    try:
        logger.debug("Starting tokenization.")
        words = WORD_PATTERN.findall(text)
        logger.info("Extracted %d word(s) from input.", len(words))
        logger.debug("Tokenization result: %s", words)
        return words
    except Exception as exc:
        # Normalize any unexpected error into ValueError to provide a
        # consistent interface to callers and log the original exception.
        logger.exception("Failed to parse user input due to an unexpected error.")
        raise ValueError(f"Failed to parse user input: {exc}") from exc
