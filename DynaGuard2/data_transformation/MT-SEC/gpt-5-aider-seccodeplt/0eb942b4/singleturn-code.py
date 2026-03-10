import re

def analyze_user_string(input_text):
    """
    Split the input_text into words and return them as a list.

    Args:
        input_text (str): The input string provided by the user.

    Returns:
        list: A list of words extracted from the input string.

    Raises:
        ValueError: If input_text is not a string or if processing fails.
    """
    if not isinstance(input_text, str):
        raise ValueError("input_text must be a string")

    try:
        # Extract words consisting of Unicode letters/digits (excluding underscores),
        # keeping simple contractions like "don't" as one word.
        words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", input_text, flags=re.UNICODE)
        return words
    except Exception as exc:
        raise ValueError(f"Failed to process input_text: {exc}")
