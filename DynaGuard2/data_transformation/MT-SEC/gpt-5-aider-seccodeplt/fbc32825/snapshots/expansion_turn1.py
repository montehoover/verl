from typing import List

def split_text(text: str, delimiter: str) -> List[str]:
    """
    Splits the given text by the specified delimiter and returns a list of substrings.

    Args:
        text: The input string to split.
        delimiter: The substring delimiter to split on.

    Returns:
        A list of substrings.
    """
    return text.split(delimiter)
