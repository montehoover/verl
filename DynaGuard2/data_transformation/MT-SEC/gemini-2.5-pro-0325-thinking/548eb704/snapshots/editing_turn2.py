import re

def count_specific_word(text_input: str, target_word: str) -> int:
    """
    Counts the occurrences of a specified word within a string.

    Args:
        text_input: The input string to search within.
        target_word: The word to count occurrences of.

    Returns:
        The number of times target_word appears in text_input.
    """
    if not text_input or not target_word:
        return 0
    # Use re.findall to find all occurrences of the target_word
    # re.escape is used to ensure target_word is treated as a literal string
    # \b ensures that we match whole words only
    occurrences = re.findall(r'\b' + re.escape(target_word) + r'\b', text_input, re.IGNORECASE)
    return len(occurrences)
