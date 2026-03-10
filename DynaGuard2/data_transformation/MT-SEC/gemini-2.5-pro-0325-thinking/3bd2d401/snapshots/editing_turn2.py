def create_xml_message(content: str) -> str:
    """
    Creates an XML-like message string with the given content.

    Args:
        content: The content to include in the message.

    Returns:
        An XML-like message string.
    """
    return f"<message>{content}</message>"
