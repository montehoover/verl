from typing import List

def split_string(input_string: str) -> List[str]:
    """
    Splits a string into a list of words.

    Args:
        input_string: The string to be split.

    Returns:
        A list of words from the input string.
    """
    return input_string.split()

def clean_and_split_string(input_string: str) -> List[str]:
    """
    Cleans a string by removing leading/trailing whitespace and converting
    to lowercase, then splits it into a list of words.

    Args:
        input_string: The string to be cleaned and split.

    Returns:
        A list of cleaned words from the input string.
    """
    cleaned_string = input_string.strip().lower()
    return cleaned_string.split()
