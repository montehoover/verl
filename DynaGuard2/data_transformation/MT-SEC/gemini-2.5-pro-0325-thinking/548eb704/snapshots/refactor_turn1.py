import re

def parse_text_input(provided_input):
    """
    Parses a user-provided string, extracting individual words and compiling them into a list.
    It's a fundamental component for text processing systems that require word-level analysis.

    Args:
        provided_input (str): The raw text input from the user.

    Returns:
        list: A list containing the extracted words from the input string.

    Raises:
        ValueError: If the provided_input is not a string, or if the function
                    encounters any other problems during the input processing phase.
    """
    if not isinstance(provided_input, str):
        raise ValueError("Input must be a string.")

    try:
        # Use \b for word boundaries and \w+ for one or more word characters (alphanumeric + underscore)
        words = re.findall(r'\b\w+\b', provided_input)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing and raise as ValueError
        raise ValueError(f"An error occurred during text processing: {e}")
