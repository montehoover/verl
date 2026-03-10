def format_user_intro(username: str, bio: str, message: str) -> str:
    """
    Formats a user introduction string for a shell script.

    Args:
        username: The name of the user.
        bio: A short biography of the user.
        message: A short message for the user.

    Returns:
        A formatted string like 'echo Name: <username>, Bio: <bio>, Message: <message>'.
    """
    return f"echo Name: {username}, Bio: {bio}, Message: {message}"
