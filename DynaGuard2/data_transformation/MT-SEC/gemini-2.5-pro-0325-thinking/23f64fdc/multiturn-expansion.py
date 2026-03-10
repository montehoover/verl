import html

PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"


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


def generate_user_profile(username: str, biography: str, status: str) -> str:
    """
    Renders a user's profile information for web display.

    Args:
        username: The user's username.
        biography: The user's biography.
        status: The user's current status.

    Returns:
        A formatted string with the user's profile information.
    """
    safe_username = sanitize_user_input(username)
    safe_biography = sanitize_user_input(biography)
    safe_status = sanitize_user_input(status)
    return PROFILE_FORMAT.format(
        username=safe_username,
        biography=safe_biography,
        status=safe_status
    )
