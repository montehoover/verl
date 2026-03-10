def extract_words(text: str) -> list[str]:
    """
    Extracts words from a given string.

    Args:
        text: The input string.

    Returns:
        A list of words found in the string.
    """
    words = text.split()
    return words
