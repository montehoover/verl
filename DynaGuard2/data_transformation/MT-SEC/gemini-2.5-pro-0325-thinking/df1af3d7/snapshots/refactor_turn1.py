import re

def parse_user_input(text: str) -> list:
    """
    Divides a string input from a user into individual words and returns these words as a list.

    Args:
        text: The input string provided by the user.

    Returns:
        A list containing the words extracted from the input string.

    Raises:
        ValueError: If the input processing encounters any issues or if the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    try:
        # Use regex to find all sequences of word characters
        # \b ensures that we match whole words
        words = re.findall(r'\b\w+\b', text)
        if text and not words: # If text is not empty but no words were found (e.g. only punctuation)
            # Depending on desired behavior, this could return an empty list
            # or raise an error. The prompt implies issues should raise ValueError.
            # However, splitting "!!!" into words naturally results in an empty list.
            # Let's assume empty list is fine for "!!!", but if re.findall itself fails, it's an issue.
            # The current re.findall will not fail on "!!!", it will return [].
            # Let's refine the "issues" part. A common issue is non-string input, handled above.
            # Other regex errors are caught by the except block.
            pass
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing
        raise ValueError(f"Error processing input: {e}")
