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


def sanitize_profile_data(username: str, biography: str, status: str) -> dict[str, str]:
    """
    Sanitize raw profile inputs for safe web display.

    Args:
        username: The user's name as they've entered it.
        biography: A user-written biography that may span multiple sentences.
        status: A brief, current status message provided by the user.

    Returns:
        A dict containing sanitized strings for 'username', 'biography', and 'status'.
    """
    username_clean = _sanitize_for_web(username.strip() if isinstance(username, str) else "")
    biography_clean = _sanitize_for_web(biography if isinstance(biography, str) else "")
    status_clean = _sanitize_for_web(status if isinstance(status, str) else "")

    return {
        "username": username_clean,
        "biography": biography_clean,
        "status": status_clean,
    }


def format_profile(sanitized: dict[str, str]) -> str:
    """
    Format sanitized profile data into a display string.

    Args:
        sanitized: A dict with keys 'username', 'biography', and 'status' containing sanitized text.

    Returns:
        A formatted string ready for web display.
    """
    return PROFILE_FORMAT.format(
        username=sanitized.get("username", ""),
        biography=sanitized.get("biography", ""),
        status=sanitized.get("status", ""),
    )


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
    sanitized = sanitize_profile_data(username, biography, status)
    return format_profile(sanitized)
