import re

def transform_user_text(text_input):
    """
    Parses a user-provided string, extracting individual words and compiling them into a list.

    Args:
        text_input: The raw text input from the user.

    Returns:
        list: Containing the extracted words from the input string.

    Raises:
        ValueError: If the input is not a string or if other processing problems occur.
    """
    if not isinstance(text_input, str):
        raise ValueError("Input must be a string.")
    
    try:
        # Use \b to ensure matching whole words
        # \w+ matches one or more word characters (letters, numbers, underscore)
        words = re.findall(r'\b\w+\b', text_input)
        return words
    except Exception as e:
        # Catch any other unexpected errors during regex processing
        raise ValueError(f"Error processing input: {e}")
