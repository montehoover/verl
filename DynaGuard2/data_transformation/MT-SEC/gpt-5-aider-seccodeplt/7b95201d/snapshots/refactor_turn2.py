"""
Utilities for parsing user-provided text into word tokens.
"""

import re


# Matches words made of word characters, allowing internal apostrophes or hyphens.
# Examples: "don't", "state-of-the-art"
WORD_PATTERN = re.compile(r"\b\w+(?:[\'-]\w+)*\b")


def transform_user_text(text_input):
    """
    Parse a user-provided string, extracting individual words and compiling them into a list.

    This function uses a regular expression to identify words, including those with
    internal apostrophes or hyphens (e.g., "don't", "state-of-the-art").

    Args:
        text_input (str): The raw text input from the user.

    Returns:
        list: A list containing the extracted words from the input string.

    Raises:
        ValueError: If the input is not a string, or if an error occurs during processing.
    """
    if not isinstance(text_input, str):
        raise ValueError("text_input must be a string")

    try:
        return WORD_PATTERN.findall(text_input)
    except Exception as exc:
        raise ValueError(f"Error processing input: {exc}") from exc
