import re

def count_words(text: str) -> int:
    """
    Counts the number of words in a string, based on spaces.

    Args:
        text: The input string.

    Returns:
        The number of words in the string.
    """
    if not text:
        return 0
    words = text.split()
    return len(words)


def extract_words(text: str) -> list[str]:
    """
    Extracts words from a string, based on spaces.

    Args:
        text: The input string.

    Returns:
        A list of words from the string.
    """
    if not text:
        return []
    return text.split()


def analyze_user_string(input_text: str) -> list[str]:
    """
    Analyzes an input string and returns a list of words.

    Args:
        input_text: The string to analyze.

    Returns:
        A list of words extracted from the input string.

    Raises:
        ValueError: If input_text is not a string or is empty.
    """
    if not isinstance(input_text, str):
        raise ValueError("Input must be a string.")
    if not input_text.strip():
        raise ValueError("Input string cannot be empty or contain only whitespace.")

    # Use regex to find all sequences of word characters
    words = re.findall(r'\b\w+\b', input_text.lower())
    return words
