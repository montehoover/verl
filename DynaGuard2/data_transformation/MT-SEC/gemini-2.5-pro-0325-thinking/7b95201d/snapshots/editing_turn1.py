import re

def count_words(text_content: str) -> int:
    """
    Counts the number of words in a given string.

    Args:
        text_content: The string to count words from.

    Returns:
        An integer representing the total count of words.
        Returns 0 if the string is empty or contains no words.
    """
    if not text_content:
        return 0
    # Find all sequences of word characters
    words = re.findall(r'\b\w+\b', text_content)
    return len(words)
