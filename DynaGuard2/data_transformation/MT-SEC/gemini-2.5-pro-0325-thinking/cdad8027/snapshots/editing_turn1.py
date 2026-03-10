def count_words(text: str) -> int:
    """
    Counts the number of words in a given string.
    Words are assumed to be separated by spaces.

    Args:
        text: The input string.

    Returns:
        The number of words in the string.
    """
    if not text:
        return 0
    words = text.split(' ')
    # Filter out empty strings that may result from multiple spaces
    return len([word for word in words if word])
