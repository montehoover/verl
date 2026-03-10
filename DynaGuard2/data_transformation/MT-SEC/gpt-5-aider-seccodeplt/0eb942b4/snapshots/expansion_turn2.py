from typing import List

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
