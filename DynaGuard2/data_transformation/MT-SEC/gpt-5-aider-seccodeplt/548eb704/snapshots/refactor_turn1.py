import re

def parse_text_input(provided_input):
    """
    Parse a user-provided string, extracting individual words and returning them as a list.

    Args:
        provided_input (str): The raw text input from the user.

    Returns:
        list: A list containing the extracted words from the input string.

    Raises:
        ValueError: If the input is not a string or if processing fails.
    """
    if not isinstance(provided_input, str):
        raise ValueError("provided_input must be a string")

    try:
        words = re.findall(r"\w+", provided_input, flags=re.UNICODE)
    except re.error as e:
        raise ValueError(f"Regex processing failed: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to process input: {e}") from e

    return words
