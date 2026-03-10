def format_user_message(username: str, message: str) -> str:
    """
    Format a user's name and message for display.

    Args:
        username (str): The user's name.
        message (str): The user's message.

    Returns:
        str: A formatted string like 'User: <username>, Message: <message>'.
    """
    return f"User: {username}, Message: {message}"
