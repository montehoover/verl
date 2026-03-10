def format_user_message_with_title(username: str, title: str, message: str) -> str:
    """
    Formats a user's name, message title, and message for display.

    Args:
        username: The username of the sender.
        title: The title of the message.
        message: The message content.

    Returns:
        A string formatted as 'Title: <title>, User: <username>, Message: <message>'.
    """
    return f"Title: {title}, User: {username}, Message: {message}"
