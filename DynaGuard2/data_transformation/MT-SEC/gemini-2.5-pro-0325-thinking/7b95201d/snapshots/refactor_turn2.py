import re


def transform_user_text(text_input: str) -> list[str]:
    """
    Parses a user-provided string, extracting individual words and compiling them into a list.

    Args:
        text_input: The raw text input from the user.

    Returns:
        A list containing the extracted words from the input string.

    Raises:
        ValueError: If the input is not a string or if any other problem occurs
                    during the input processing phase.
    """
    if not isinstance(text_input, str):
        raise ValueError("Input must be a string.")
    
    try:
        # Use \w+ to match sequences of word characters (letters, numbers, and underscore)
        words = re.findall(r'\w+', text_input)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing
        raise ValueError(f"Error processing input: {e}")
