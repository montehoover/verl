from html import escape

PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"


def _sanitize_for_web(text: str) -> str:
    """
    Sanitize user-provided text for safe web display.

    - Escapes HTML special characters to prevent injection.
    - Normalizes newlines and converts them to <br> for display.
    """
    if text is None:
        text = ""
    # Normalize newlines to \n
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Escape HTML
    escaped = escape(normalized, quote=True)
    # Preserve intended line breaks
    return escaped.replace("\n", "<br>")


def generate_user_profile(username: str, biography: str, status: str) -> str:
    """
    Render a user's profile information for web display.

    Args:
        username: The user's name as they've entered it.
        biography: A user-written biography that may span multiple sentences.
        status: A brief, current status message provided by the user.

    Returns:
        A formatted string containing the sanitized user profile information
        ready for web display.
    """
    username_clean = _sanitize_for_web(username.strip() if isinstance(username, str) else "")
    biography_clean = _sanitize_for_web(biography if isinstance(biography, str) else "")
    status_clean = _sanitize_for_web(status if isinstance(status, str) else "")

    return PROFILE_FORMAT.format(
        username=username_clean,
        biography=biography_clean,
        status=status_clean,
    )
