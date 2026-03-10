from html import escape

# Pre-defined template for formatting user profiles
PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"


def _sanitize_text(value: str) -> str:
    """
    Sanitize a user-provided string for safe HTML display.
    - Converts None to empty string.
    - Escapes HTML special characters.
    - Converts newline characters to <br> to preserve formatting.
    - Trims leading and trailing whitespace.
    """
    if value is None:
        value = ""
    text = str(value)
    # Normalize newlines to \n before replacement
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Escape HTML special chars
    text = escape(text, quote=True)
    # Preserve user-intended line breaks
    text = text.replace("\n", "<br>")
    # Trim surrounding whitespace (after conversion)
    return text.strip()


def display_user_profile(username: str, about: str, current_status: str) -> str:
    """
    Generate a sanitized and formatted user profile block for web display.

    Args:
        username: The user's full name.
        about: A multi-sentence biography provided by the user.
        current_status: A brief, current status update from the user.

    Returns:
        A string containing sanitized and formatted profile information,
        using the PROFILE_FORMAT template.
    """
    safe_username = _sanitize_text(username)
    safe_about = _sanitize_text(about)
    safe_status = _sanitize_text(current_status)

    return PROFILE_FORMAT.format(
        username=safe_username,
        about=safe_about,
        current_status=safe_status,
    )
