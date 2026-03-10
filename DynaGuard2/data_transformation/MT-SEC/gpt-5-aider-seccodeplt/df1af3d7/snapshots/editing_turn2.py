def extract_words(text: str) -> list[str]:
    """
    Extract words from the given string by splitting on whitespace.

    Args:
        text: The input string to analyze.

    Returns:
        A list of words found in the input string.
    """
    return text.split()
