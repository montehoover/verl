from typing import Optional
import re

def validate_and_cleanup(text_input: str) -> Optional[str]:
    """
    Strip leading and trailing whitespace from the input string.
    Return the cleaned string if non-empty; otherwise return None.
    """
    if not isinstance(text_input, str):
        return None
    cleaned = text_input.strip()
    return cleaned if cleaned != "" else None

def normalize_text_case(cleaned_text: str) -> str:
    """
    Convert all characters in the input string to lowercase without altering its structure.
    """
    return cleaned_text.lower()

def parse_text_input(provided_input: str):
    """
    Use regular expressions to extract individual words from the provided input string.
    Returns a list of words. Raises ValueError on processing errors.
    """
    if not isinstance(provided_input, str):
        raise ValueError("provided_input must be a string")
    try:
        # Extract "words" using Unicode-aware word characters
        words = re.findall(r"\b\w+\b", provided_input, flags=re.UNICODE)
        return words
    except re.error as e:
        raise ValueError(f"Regex processing failed: {e}") from e
