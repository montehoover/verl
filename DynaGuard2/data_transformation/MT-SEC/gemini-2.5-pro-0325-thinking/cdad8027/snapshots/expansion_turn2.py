import re
from typing import AnyStr

def process_input_text(txt: AnyStr) -> AnyStr:
    """
    Converts the input text to a standardized format (lowercase).

    Args:
        txt: The input string.

    Returns:
        The lowercased string.
    """
    if isinstance(txt, str):
        return txt.lower()
    elif isinstance(txt, bytes):
        return txt.lower()
    else:
        # Or raise a TypeError, depending on desired handling for other types
        return txt

def clean_text(txt: str) -> str:
    """
    Removes punctuation and special characters from the input text.

    Args:
        txt: The input string.

    Returns:
        The cleaned string.
    """
    # Remove punctuation and special characters using regex
    # This pattern keeps alphanumeric characters and spaces
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
    return cleaned_text
