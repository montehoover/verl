import html

PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"

def create_user_info_html(username: str, user_info: str) -> str:
    """
    Create an HTML string containing user details.

    Args:
        username: The user's name.
        user_info: The user's information.

    Returns:
        A string in the format: <div><h2>{username}</h2><p>{user_info}</p></div>
    """
    return f"<div><h2>{username}</h2><p>{user_info}</p></div>"

def sanitize_user_input(text: str) -> str:
    """
    Sanitize a string for safe HTML display by escaping special characters.

    Args:
        text: The input string to sanitize.

    Returns:
        A sanitized string with HTML-sensitive characters escaped.
    """
    return html.escape(text, quote=True)

def generate_user_profile(username: str, biography: str, status: str) -> str:
    """
    Render a user's profile for web display using PROFILE_FORMAT.

    Args:
        username: The user's name.
        biography: The user's biography.
        status: The user's status message.

    Returns:
        A formatted, HTML-safe string following PROFILE_FORMAT.
    """
    safe_username = sanitize_user_input(username)
    safe_biography = sanitize_user_input(biography)
    safe_status = sanitize_user_input(status)

    return PROFILE_FORMAT.format(
        username=safe_username,
        biography=safe_biography,
        status=safe_status
    )
