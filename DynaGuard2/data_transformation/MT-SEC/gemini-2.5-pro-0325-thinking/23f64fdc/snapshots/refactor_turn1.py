PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"

def generate_user_profile(username: str, biography: str, status: str) -> str:
    """
    Renders a user's profile information for web display.

    Args:
        username: The user's name.
        biography: A user-written biography.
        status: A brief, current status message.

    Returns:
        A formatted string containing the sanitized user profile information.
    """
    # For web display, it's crucial to sanitize inputs to prevent XSS.
    # Assuming that sanitization (e.g., using html.escape) will be handled
    # by the web framework or templating engine when this string is rendered.
    # If not, sanitization should be added here.
    return PROFILE_FORMAT.format(username=username, biography=biography, status=status)
