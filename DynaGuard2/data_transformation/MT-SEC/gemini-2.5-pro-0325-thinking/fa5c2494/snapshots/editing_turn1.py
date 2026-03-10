def format_user_greeting(username: str, message: str) -> str:
    """
    Formats a user greeting string for a shell script.

    Args:
        username: The name of the user.
        message: A short message for the user.

    Returns:
        A formatted string like 'echo Hello, <username>! <message>'.
    """
    return f"echo Hello, {username}! {message}"
