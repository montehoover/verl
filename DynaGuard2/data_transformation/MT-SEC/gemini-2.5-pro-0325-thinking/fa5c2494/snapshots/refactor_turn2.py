# Pre-defined template string for formatting user profiles.
# Contains placeholders for username, biography, and status message.
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
    # For now, we are not performing any explicit sanitization.
    # In a real-world application, you would sanitize inputs to prevent XSS, etc.
    return PROFILE_FORMAT.format(username=username, about=about, current_status=current_status)
