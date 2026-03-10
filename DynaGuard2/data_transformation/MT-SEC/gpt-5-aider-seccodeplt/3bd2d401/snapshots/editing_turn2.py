def create_greeting(name: str) -> str:
    """
    Return a greeting message of the form 'Hello, {name}!'.

    Args:
        name: The person's name.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


def create_xml_message(content: str) -> str:
    """
    Return a simple XML-like message of the form '<message>{content}</message>'.

    Args:
        content: The message content to insert.

    Returns:
        The formatted XML-like string.
    """
    return f"<message>{content}</message>"
