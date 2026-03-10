from typing import Optional

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
