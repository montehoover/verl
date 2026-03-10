def text_to_paragraph(text: str) -> str:
    """
    Wraps the given text string in HTML <p> tags.

    Args:
        text: The string to wrap.

    Returns:
        The text string wrapped in <p> tags.
    """
    return f"<p>{text}</p>"
