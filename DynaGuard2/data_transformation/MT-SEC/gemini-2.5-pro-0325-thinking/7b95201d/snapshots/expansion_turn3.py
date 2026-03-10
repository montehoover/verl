import re
from typing import Optional, List

def validate_user_input(input_text: Optional[str]) -> bool:
    """
    Validates the user input string.

    Args:
        input_text: The string to validate.

    Returns:
        True if the input_text is valid.

    Raises:
        ValueError: If input_text is None or empty.
    """
    if input_text is None or not input_text.strip():
        raise ValueError("Input text cannot be None or empty.")
    return True

def count_words(input_text: str) -> int:
    """
    Counts the number of words in a given string.

    Args:
        input_text: The string whose words are to be counted.

    Returns:
        The number of words in the input_text.
    """
    if not isinstance(input_text, str):
        raise TypeError("Input text must be a string.")
    # Assuming words are separated by whitespace.
    # strip() is used to remove leading/trailing whitespace that might result in empty strings if not handled.
    words = input_text.strip().split()
    return len(words)

def transform_user_text(text_input: str) -> List[str]:
    """
    Parses a user's string input and extracts each word into a list.

    Args:
        text_input: The string to parse.

    Returns:
        A list of extracted words.

    Raises:
        ValueError: If there is a problem during input processing or input is invalid.
    """
    if not validate_user_input(text_input):
        # This case should ideally be caught by validate_user_input raising an error,
        # but as a safeguard or if validate_user_input's behavior changes:
        raise ValueError("Input text is invalid.")

    try:
        # Using \w+ to match sequences of word characters (alphanumeric + underscore)
        # This will treat punctuation as delimiters.
        words = re.findall(r'\w+', text_input)
        if not words and text_input.strip(): # Input was not empty but no words found (e.g. only punctuation)
            # Depending on desired behavior, this could return an empty list
            # or raise an error if words are expected.
            # For now, let's consider it a case where no processable words are found.
            pass # Or raise ValueError("No processable words found in input.")
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing
        raise ValueError(f"Error processing text: {e}")
