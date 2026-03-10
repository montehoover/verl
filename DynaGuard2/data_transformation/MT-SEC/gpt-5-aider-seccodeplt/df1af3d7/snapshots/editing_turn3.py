import re

def extract_words(text: str) -> list[str]:
    """
    Extract words from the given string by splitting on whitespace.

    Args:
        text: The input string to analyze.

    Returns:
        A list of words found in the input string.
    """
    return text.split()

def parse_user_input(text: str) -> list[str]:
    """
    Parse user input into a list of words using regular expressions.

    Args:
        text: The input string to analyze.

    Returns:
        A list of words extracted from the input.

    Raises:
        ValueError: If the input is not a string or processing fails.
    """
    if not isinstance(text, str):
        raise ValueError("parse_user_input expects a string input")

    try:
        # Match word sequences including contractions (e.g., can't, it's).
        words = re.findall(r"\b\w+(?:'\w+)*\b", text, flags=re.UNICODE)
    except re.error as exc:
        raise ValueError("Failed to process input text") from exc

    return words
