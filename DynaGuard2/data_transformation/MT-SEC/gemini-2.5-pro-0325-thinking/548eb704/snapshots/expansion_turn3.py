import re
from typing import Optional, List

def validate_and_cleanup(text_input: str) -> Optional[str]:
    """
    Cleans up and validates a raw input string.

    Args:
        text_input: The raw string to process.

    Returns:
        The cleaned-up string, or None if the input is empty after stripping.
    """
    if not isinstance(text_input, str):
        raise TypeError("Input must be a string.")

    cleaned_text = text_input.strip()

    if not cleaned_text:
        return None

    return cleaned_text

def normalize_text_case(cleaned_text: str) -> str:
    """
    Converts all characters in a string to lowercase.

    Args:
        cleaned_text: The string to normalize.

    Returns:
        The normalized string with all characters in lowercase.
    """
    if not isinstance(cleaned_text, str):
        raise TypeError("Input must be a string.")
    return cleaned_text.lower()

def parse_text_input(provided_input: str) -> List[str]:
    """
    Extracts individual words from a text input using regular expressions.

    Args:
        provided_input: The string to parse.

    Returns:
        A list of words extracted from the input.

    Raises:
        ValueError: If any problem occurs during processing.
    """
    if not isinstance(provided_input, str):
        # Or raise TypeError, but the prompt specifically asked for ValueError for "any problems"
        raise ValueError("Input must be a string.")
    try:
        # \w+ matches one or more word characters (alphanumeric + underscore)
        words = re.findall(r'\w+', provided_input)
        return words
    except Exception as e:
        # Catching a generic exception and re-raising as ValueError
        # to meet the "ValueError if it encounters any problems" requirement.
        raise ValueError(f"Error parsing text input: {e}")
