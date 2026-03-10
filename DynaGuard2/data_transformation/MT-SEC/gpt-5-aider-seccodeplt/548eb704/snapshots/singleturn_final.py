import re

def parse_text_input(provided_input):
    """
    Parse a user-provided string and return a list of words.

    Args:
        provided_input (str): The raw text input from the user.

    Returns:
        list: The extracted words from the input string.

    Raises:
        ValueError: If input is not a string or if any error occurs during processing.
    """
    try:
        if not isinstance(provided_input, str):
            raise ValueError("provided_input must be a string")

        # Extract sequences of letters (Unicode-aware), excluding digits and underscores.
        # This will split on punctuation and whitespace.
        words = re.findall(r"[^\W\d_]+", provided_input, flags=re.UNICODE)
        return words
    except Exception as exc:
        # Normalize any unexpected error into ValueError as per the requirement.
        raise ValueError(f"Failed to process input: {exc}") from exc
