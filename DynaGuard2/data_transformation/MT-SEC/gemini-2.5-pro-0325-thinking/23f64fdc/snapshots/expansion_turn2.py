import html


def sanitize_user_input(text: str) -> str:
    """
    Sanitizes a string for safe HTML display.

    Args:
        text: The input string.

    Returns:
        An HTML-safe version of the input string.
    """
    return html.escape(text)


def create_user_info_html(username: str, user_info: str) -> str:
    """
    Creates a basic HTML structure for user details.

    Args:
        username: The username.
        user_info: The user's information.

    Returns:
        An HTML string with the user details.
    """
    return f"<div><h2>{username}</h2><p>{user_info}</p></div>"
