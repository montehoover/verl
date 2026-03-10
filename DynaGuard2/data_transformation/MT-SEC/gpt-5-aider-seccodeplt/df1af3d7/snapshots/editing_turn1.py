from typing import Any


def count_words(text: str) -> int:
    """
    Count the number of words in the given string.

    A word is defined as a sequence of non-whitespace characters separated by
    whitespace, as determined by str.split().

    Args:
        text: The input string to analyze.

    Returns:
        The total number of words in the input string.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("count_words expects a string")
    return len(text.split())
