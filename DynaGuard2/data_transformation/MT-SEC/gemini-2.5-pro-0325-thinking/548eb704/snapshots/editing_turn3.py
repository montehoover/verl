import re
from typing import List

def parse_text_input(provided_input: str) -> List[str]:
    """
    Extracts individual words from a user-provided string.

    Args:
        provided_input: The raw text input.

    Returns:
        A list of extracted words.

    Raises:
        ValueError: If the input is not a string or if there are
                    any issues during processing.
    """
    if not isinstance(provided_input, str):
        raise ValueError("Input must be a string.")

    try:
        # Use re.findall to find all sequences of word characters.
        # \b ensures that we match whole words.
        words = re.findall(r'\b\w+\b', provided_input)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing.
        raise ValueError(f"Error processing text input: {e}")
