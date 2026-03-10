import html


def sanitize_html(text: str) -> str:
    """
    Escapes special HTML characters in the given text.

    Args:
        text: The string to be sanitized.

    Returns:
        A string with special HTML characters escaped.
    """
    return html.escape(text)


def create_html_paragraph(text: str) -> str:
    """
    Wraps the given text in an HTML <p> element.

    Args:
        text: The string to be wrapped.

    Returns:
        An HTML string with the text wrapped in a <p> element.
    """
    return f"<p>{text}</p>"
