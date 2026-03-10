def format_user_message(username: str, message: str) -> str:
    """
    Formats a user's name and message for display.

    Args:
        username: The username of the sender.
        message: The message content.

    Returns:
        A string formatted as 'User: <username>, Message: <message>'.
    """
    return f"User: {username}, Message: {message}"
