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
