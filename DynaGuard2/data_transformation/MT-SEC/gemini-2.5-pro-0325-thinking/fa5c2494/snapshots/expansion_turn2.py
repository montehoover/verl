import html

def sanitize_user_input(text: str) -> str:
    """
    Sanitizes a string for safe HTML display.

    Args:
        text: The string to sanitize.

    Returns:
        An HTML-safe version of the input string.
    """
    return html.escape(text)

def create_user_info_html(username: str, user_info: str) -> str:
    """
    Generates an HTML string for a user profile.

    Args:
        username: The user's username.
        user_info: The user's information.

    Returns:
        An HTML string representing the user's profile.
    """
    return f"<div><h2>{username}</h2><p>{user_info}</p></div>"
