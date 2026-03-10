import re
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

def analyze_user_string(input_string: str) -> List[str]:
    """
    Analyzes a user-provided string, splits it into words, and returns them.

    Args:
        input_string: The string to be analyzed.

    Returns:
        A list of words from the input string.

    Raises:
        ValueError: If the input is not a string or if processing fails
                    (e.g., results in an empty list of words after cleaning).
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string.")

    try:
        # Utilize the existing cleaning and splitting function
        words = clean_and_split_string(input_string)
        if not words:
            # Consider it an issue if no words are found after cleaning
            raise ValueError("Processing resulted in no words. Input might be empty or only whitespace.")
        return words
    except Exception as e:
        # Catch any other unexpected errors during processing
        raise ValueError(f"Error processing string: {e}")
