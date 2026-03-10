import re
from typing import List

def transform_user_text(text_input: str) -> List[str]:
    """
    Extracts each word from a given string into a list.

    Args:
        text_input: The string to extract words from.

    Returns:
        A list of all words found in the text_input.

    Raises:
        ValueError: If there's any issue during processing, including if text_input is not a string.
    """
    if not isinstance(text_input, str):
        raise ValueError("Input must be a string.")
    try:
        # Find all sequences of word characters
        words = re.findall(r'\b\w+\b', text_input)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing or other operations
        raise ValueError(f"Error processing text: {e}")
