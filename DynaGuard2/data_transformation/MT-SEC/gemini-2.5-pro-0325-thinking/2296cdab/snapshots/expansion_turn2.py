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

def wrap_with_tag(tag: str, content: str) -> str:
    """
    Wraps the given content string with the specified HTML tag.

    Args:
        tag: The HTML tag to use (e.g., 'p', 'div').
        content: The string content to be wrapped.

    Returns:
        A string with the content wrapped in HTML tags.
    """
    return f"<{tag}>{content}</{tag}>"
