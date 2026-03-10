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
