def count_words(text: str) -> int:
    """
    Counts the number of words in a given string.

    Args:
        text: The input string.

    Returns:
        The total number of words in the string.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    if not text:
        return 0
    words = text.split()
    return len(words)
