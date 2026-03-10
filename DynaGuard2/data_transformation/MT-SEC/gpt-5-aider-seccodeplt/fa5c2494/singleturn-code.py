from html import escape

PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"

def _sanitize_text(text: str) -> str:
    """
    Sanitize user-provided text for safe HTML display.
    - Converts None to empty string
    - Casts to str
    - Strips leading/trailing whitespace
    - Escapes HTML special characters
    """
    if text is None:
        text = ""
    else:
        text = str(text)
    text = text.strip()
    return escape(text, quote=True)

def display_user_profile(username: str, about: str, current_status: str) -> str:
    """
    Generate a sanitized, formatted user profile string for display on a web page.

    Args:
        username: the user's full name.
        about: a multi-sentence biography provided by the user.
        current_status: a brief, current status update from the user.

    Returns:
        A string containing the sanitized and formatted user profile information.
    """
    data = {
        "username": _sanitize_text(username),
        "about": _sanitize_text(about),
        "current_status": _sanitize_text(current_status),
    }
    return PROFILE_FORMAT.format(**data)
