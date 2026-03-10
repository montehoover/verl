import re

def parse_user_input(text: str) -> list[str]:
    """
    Extracts words from the input string and handles potential issues.

    Args:
        text: The input string.

    Returns:
        A list of words extracted from the string.

    Raises:
        ValueError: If any issue occurs during processing or if the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    
    try:
        # Use regex to find all words. \b ensures word boundaries.
        words = re.findall(r'\b\w+\b', text)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing or other operations
        raise ValueError(f"Error processing input: {e}")
