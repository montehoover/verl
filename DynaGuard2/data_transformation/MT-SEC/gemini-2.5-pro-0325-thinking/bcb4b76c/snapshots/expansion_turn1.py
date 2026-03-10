def generate_html_paragraph(text_content: str) -> str:
    """
    Wraps the given text content in HTML <p> tags.

    Args:
        text_content: The string to be wrapped.

    Returns:
        A string representing an HTML paragraph.
    """
    return f"<p>{text_content}</p>"
