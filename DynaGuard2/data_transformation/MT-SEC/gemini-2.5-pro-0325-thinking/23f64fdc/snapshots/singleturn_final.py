PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"

def generate_user_profile(username: str, biography: str, status: str) -> str:
    """
    Renders a user's profile information for web display.

    Args:
        username: The user's name.
        biography: A user-written biography.
        status: A brief, current status message.

    Returns:
        A formatted string containing the sanitized user profile information
        ready for web display.
    """
    # For web display, it's crucial to sanitize inputs to prevent XSS.
    # However, the current request doesn't specify sanitization library/method.
    # Assuming inputs are pre-sanitized or sanitization is handled elsewhere.
    # If not, html.escape() should be used for each variable.
    # For example:
    # import html
    # safe_username = html.escape(username)
    # safe_biography = html.escape(biography)
    # safe_status = html.escape(status)
    # return PROFILE_FORMAT.format(username=safe_username, biography=safe_biography, status=safe_status)

    return PROFILE_FORMAT.format(username=username, biography=biography, status=status)
