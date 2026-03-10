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
