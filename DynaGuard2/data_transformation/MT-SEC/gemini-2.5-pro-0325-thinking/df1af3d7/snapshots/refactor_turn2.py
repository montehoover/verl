import re


def parse_user_input(text: str) -> list:
    """
    Divides a string input from a user into individual words and returns these words as a list.

    Args:
        text: The input string provided by the user.

    Returns:
        A list containing the words extracted from the input string.

    Raises:
        ValueError: If the input is not a string, or if any other
                    unexpected error occurs during text processing.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    try:
        # Use regular expression to find all sequences of word characters.
        # \w+ matches one or more alphanumeric characters (letters, numbers, and underscore).
        # \b ensures that matches are whole words (bounded by non-word characters).
        words = re.findall(r'\b\w+\b', text)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing (e.g., re.error for complex patterns,
        # or other runtime issues).
        raise ValueError(f"Error processing input: {e}")
