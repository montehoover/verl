import re

def count_words(text: str) -> int:
    """
    Counts the number of words in a given text.

    Args:
        text: The input string.

    Returns:
        The total number of words in the text.
    """
    if not text:
        return 0
    # Use re.findall to find all sequences of word characters
    words = re.findall(r'\b\w+\b', text)
    return len(words)
