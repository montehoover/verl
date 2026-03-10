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
