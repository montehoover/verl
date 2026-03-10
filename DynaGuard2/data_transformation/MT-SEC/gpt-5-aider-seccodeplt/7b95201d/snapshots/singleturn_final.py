import re

def transform_user_text(text_input):
    """
    Parse a user-provided string into a list of words.

    Args:
        text_input (str): The raw text input from the user.

    Returns:
        list: A list containing the extracted words from the input string.

    Raises:
        ValueError: If the input is not a string or if processing fails.
    """
    if not isinstance(text_input, str):
        raise ValueError("text_input must be a string")

    try:
        # Extract words consisting of letters/numbers, allowing an internal apostrophe (e.g., don't).
        # Punctuation and other symbols are ignored.
        pattern = r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?"
        words = re.findall(pattern, text_input)
        return words
    except Exception as exc:
        # Wrap any unexpected error as a ValueError to meet the contract.
        raise ValueError(f"Failed to process input text: {exc}") from exc
