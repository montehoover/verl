"""
Utilities for parsing user-provided text into word tokens.
"""

import logging
import re


# Module-level logger for this utility module.
logger = logging.getLogger(__name__)

# Matches words made of word characters, allowing internal apostrophes or hyphens.
# Examples: "don't", "state-of-the-art"
WORD_PATTERN = re.compile(r"\b\w+(?:[\'-]\w+)*\b")


def transform_user_text(text_input):
    """
    Parse a user-provided string, extracting individual words and compiling them into a list.

    This function uses a regular expression to identify words, including those with
    internal apostrophes or hyphens (e.g., "don't", "state-of-the-art"). The function
    logs the input received and the resulting list of extracted words.

    Args:
        text_input (str): The raw text input from the user.

    Returns:
        list: A list containing the extracted words from the input string.

    Raises:
        ValueError: If the input is not a string, or if an error occurs during processing.
    """
    if not isinstance(text_input, str):
        raise ValueError("text_input must be a string")

    logger.info("transform_user_text input: %r", text_input)

    if text_input == "":
        logger.info("transform_user_text output: []")
        return []

    try:
        words = WORD_PATTERN.findall(text_input)
        logger.info("transform_user_text output: %s", words)
        return words
    except Exception as exc:
        logger.exception("Error processing input in transform_user_text")
        raise ValueError(f"Error processing input: {exc}") from exc
