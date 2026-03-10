def format_user_message_with_title(username: str, title: str, message: str) -> str:
    """
    Format a title, user's name, and message for display.

    Args:
        username (str): The user's name.
        title (str): The message title.
        message (str): The user's message.

    Returns:
        str: A formatted string like 'Title: <title>, User: <username>, Message: <message>'.
    """
    return f"Title: {title}, User: {username}, Message: {message}"
