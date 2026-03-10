import html

PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"

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

def display_user_profile(username: str, about: str, current_status: str) -> str:
    """
    Generates a formatted HTML string for a user profile, sanitizing input.

    Args:
        username: The user's username.
        about: The user's biography.
        current_status: The user's current status.

    Returns:
        A formatted and sanitized HTML string representing the user's profile.
    """
    s_username = sanitize_user_input(username)
    s_about = sanitize_user_input(about)
    s_current_status = sanitize_user_input(current_status)
    return PROFILE_FORMAT.format(username=s_username, about=s_about, current_status=s_current_status)
