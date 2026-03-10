from typing import Optional

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
