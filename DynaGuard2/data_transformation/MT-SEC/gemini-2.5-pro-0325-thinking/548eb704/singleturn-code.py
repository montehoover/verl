import re

def parse_text_input(provided_input):
    """
    Parses a user-provided string, extracting individual words and compiling them into a list.

    Args:
        provided_input: The raw text input from the user.

    Returns:
        A list containing the extracted words from the input string.

    Raises:
        ValueError: If the function encounters any problems during the input processing phase.
    """
    if not isinstance(provided_input, str):
        raise ValueError("Input must be a string.")
    
    try:
        # Use \w+ to match sequences of word characters (letters, numbers, and underscore)
        # Use \b to ensure that we match whole words
        words = re.findall(r'\b\w+\b', provided_input)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing
        raise ValueError(f"Error processing input: {e}")
