from html import escape

# Predefined template for rendering the user profile
PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"


def _sanitize_for_html(text: str) -> str:
    """
    Escape HTML-sensitive characters and convert newlines to <br> for web display.
    Ensures None is treated as an empty string and trims surrounding whitespace.
    """
    if text is None:
        text = ""
    # Escape HTML special characters
    sanitized = escape(str(text), quote=True)
    # Normalize newlines and convert to <br> for web display
    sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    # Trim surrounding whitespace
    return sanitized.strip()


def generate_user_profile(username: str, biography: str, status: str) -> str:
    """
    Render a user's profile information for web display.

    Args:
        username: The user's name as they've entered it.
        biography: A user-written biography that may span multiple sentences (and lines).
        status: A brief, current status message provided by the user.

    Returns:
        A formatted string containing the sanitized user profile information ready for web display.
    """
    safe_username = _sanitize_for_html(username)
    safe_biography = _sanitize_for_html(biography)
    safe_status = _sanitize_for_html(status)

    return PROFILE_FORMAT.format(
        username=safe_username,
        biography=safe_biography,
        status=safe_status,
    )
