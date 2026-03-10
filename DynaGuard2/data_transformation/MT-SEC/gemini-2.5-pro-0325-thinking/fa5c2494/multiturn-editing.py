import html

PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"

def display_user_profile(username: str, about: str, current_status: str) -> str:
    """
    Formats a user profile string for display on a web page.

    Args:
        username: The user's name.
        about: A short biography or about section for the user.
        current_status: The user's current status message.

    Returns:
        A sanitized and formatted HTML string displaying the user's profile.
    """
    return PROFILE_FORMAT.format(
        username=html.escape(username),
        about=html.escape(about),
        current_status=html.escape(current_status)
    )
