import html

def sanitize_text(input_string: str) -> str:
    """
    Escapes HTML special characters in a string.

    Args:
        input_string: The string to sanitize.

    Returns:
        The sanitized string with HTML special characters escaped.
    """
    return html.escape(input_string)
