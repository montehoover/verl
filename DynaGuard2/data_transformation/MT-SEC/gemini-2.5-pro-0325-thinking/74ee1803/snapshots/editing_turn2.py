def html_format_message(title: str, content: str) -> str:
    """
    Formats a message with a title and content for HTML display.

    Args:
        title: The title of the message.
        content: The content of the message.

    Returns:
        An HTML formatted string.
    """
    return f"<h1>{title}</h1><p>{content}</p>"
