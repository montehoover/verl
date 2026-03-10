def is_valid_string(input_string: str) -> bool:
    """
    Checks if a given string is non-empty and doesn't contain any spaces.

    Args:
        input_string: The string to validate.

    Returns:
        True if the string is valid, False otherwise.
    """
    if not input_string:  # Check if the string is empty
        return False
    if ' ' in input_string:  # Check if the string contains any spaces
        return False
    return True
