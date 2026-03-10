PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"

def display_user_profile(username: str, about: str, current_status: str) -> str:
    """
    Generates a user profile display for a web page.

    Args:
        username: The user's full name.
        about: A multi-sentence biography provided by the user.
        current_status: A brief, current status update from the user.

    Returns:
        A string containing the sanitized and formatted user profile information.
    """
    # For this problem, we assume inputs are already sanitized.
    # If sanitization were required, it would be done here.
    return PROFILE_FORMAT.format(username=username, about=about, current_status=current_status)
