import re

def parse_user_input(text):
    """
    Parse a user-provided string into a list of words.

    Args:
        text (str): The input string provided by the user.

    Returns:
        list: A list of words extracted from the input string.

    Raises:
        ValueError: If the input is not a string or if processing fails.
    """
    if not isinstance(text, str):
        raise ValueError("Input 'text' must be a string.")

    try:
        # Match sequences of letters/digits, allowing internal apostrophes (e.g., don't, it's)
        words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", text)
        return words
    except Exception as exc:
        raise ValueError(f"Failed to process input: {exc}") from exc
