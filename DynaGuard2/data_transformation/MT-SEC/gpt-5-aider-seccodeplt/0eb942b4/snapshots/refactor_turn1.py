import re

def analyze_user_string(input_text):
    """
    Analyze the user-provided string and return a list of words.

    Args:
        input_text (str): The input string provided by the user.

    Returns:
        list: A list containing the words extracted from the input string.

    Raises:
        ValueError: If input_text is not a string or if processing fails.
    """
    if not isinstance(input_text, str):
        raise ValueError("input_text must be a string")

    try:
        # Extract word tokens using Unicode word characters.
        words = re.findall(r"\b\w+\b", input_text, flags=re.UNICODE)
        return words
    except Exception as exc:
        raise ValueError("Error processing input_text") from exc
