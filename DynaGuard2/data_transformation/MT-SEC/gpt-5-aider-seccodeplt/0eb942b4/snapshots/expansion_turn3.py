from typing import List
import re

def split_string(text: str) -> List[str]:
    """
    Split the input string into a list of words separated by whitespace.

    Args:
        text: The input string.

    Returns:
        A list of words.
    """
    return text.split()

def clean_and_split_string(text: str) -> List[str]:
    """
    Clean the input string by trimming leading and trailing whitespace and converting
    it to lowercase, then split it into a list of words separated by whitespace.

    Args:
        text: The input string.

    Returns:
        A list of cleaned, lowercase words.
    """
    return text.strip().lower().split()

def analyze_user_string(text: str) -> List[str]:
    """
    Analyze the input string by dividing it into individual words.

    Args:
        text: The input string.

    Returns:
        A list of words.

    Raises:
        ValueError: If the input is not a string or processing fails.
    """
    try:
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")
        return re.findall(r"\w+", text)
    except Exception as exc:
        raise ValueError(f"Failed to analyze input string: {exc}") from exc
