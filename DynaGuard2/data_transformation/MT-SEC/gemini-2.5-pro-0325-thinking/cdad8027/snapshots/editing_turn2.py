from typing import List

def extract_words(text: str) -> List[str]:
    """
    Extracts words from a given string and returns them as a list.
    Words are assumed to be separated by spaces.

    Args:
        text: The input string.

    Returns:
        A list of words from the string.
    """
    if not text:
        return []
    words = text.split(' ')
    # Filter out empty strings that may result from multiple spaces
    return [word for word in words if word]
