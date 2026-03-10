import re
from typing import List

def tokenize_input_text(txt: str) -> List[str]:
    """
    Tokenizes the input string into a list of words using regular expressions.

    Args:
        txt: The input string.

    Returns:
        A list of words extracted from the string.

    Raises:
        ValueError: If any complications occur during tokenization.
    """
    if not isinstance(txt, str):
        raise ValueError("Input must be a string.")
    try:
        # Use regex to find all sequences of word characters
        words = re.findall(r'\b\w+\b', txt)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing
        raise ValueError(f"Error tokenizing input text: {e}")
