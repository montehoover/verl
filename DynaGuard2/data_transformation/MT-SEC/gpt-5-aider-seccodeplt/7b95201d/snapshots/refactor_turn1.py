import re

def transform_user_text(text_input):
    """
    Parse a user-provided string, extracting individual words and compiling them into a list.
    
    Args:
        text_input (str): The raw text input from the user.
    
    Returns:
        list: A list containing the extracted words from the input string.
    
    Raises:
        ValueError: If input is not a string or if any error occurs during processing.
    """
    if not isinstance(text_input, str):
        raise ValueError("text_input must be a string")

    try:
        # Extract words: sequences of word characters possibly containing internal apostrophes or hyphens.
        # Example: "don't", "state-of-the-art"
        pattern = r"\b\w+(?:[\'-]\w+)*\b"
        words = re.findall(pattern, text_input)
        return words
    except Exception as exc:
        raise ValueError(f"Error processing input: {exc}") from exc
