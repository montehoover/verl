def create_message_html(title: str, content: str) -> str:
    """
    Generates an HTML string with a title and content.

    Args:
        title: The title of the message (will be an <h1>).
        content: The content of the message (will be a <p>).

    Returns:
        An HTML formatted string.
    """
    html_string = f"<h1>{title}</h1>\n<p>{content}</p>"
    return html_string
