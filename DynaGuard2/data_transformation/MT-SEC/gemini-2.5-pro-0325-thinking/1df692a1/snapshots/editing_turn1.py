import html

def sanitize_input(user_input: str) -> str:
    """
    Sanitizes user input to prevent HTML or script injection attacks.

    Args:
        user_input: The string to sanitize.

    Returns:
        The sanitized string.
    """
    return html.escape(user_input)
