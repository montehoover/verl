"""
Utilities for generating sanitized HTML snippets for user profiles.

This module exposes a single public function, `display_user_profile`,
which formats user-supplied profile data into a pre-defined HTML template
while ensuring the content is safely escaped for display in a browser.
"""

from html import escape


# Pre-defined template for formatting user profiles. It contains placeholders
# for the user's name, biography, and current status.
PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"


__all__ = ["display_user_profile"]


def _sanitize_text(value: str) -> str:
    """
    Return a sanitized version of the given text for safe HTML display.

    This helper performs the following steps:
    - Converts None to an empty string.
    - Ensures the value is a string.
    - Normalizes CRLF/CR to LF newlines.
    - Escapes HTML special characters (e.g., <, >, &, ").
    - Converts LF newlines to <br> tags to preserve line breaks in HTML.
    - Trims leading and trailing whitespace.

    Args:
        value: A user-provided string (or None) to sanitize.

    Returns:
        A sanitized HTML-safe string with line breaks preserved.
    """
    if value is None:
        value = ""

    # Ensure we are dealing with text.
    text = str(value)

    # Normalize different newline conventions to a single '\n'.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Escape HTML special characters to prevent injection.
    text = escape(text, quote=True)

    # Preserve user-intended line breaks in HTML.
    text = text.replace("\n", "<br>")

    # Remove surrounding whitespace after processing.
    return text.strip()


def display_user_profile(username: str, about: str, current_status: str) -> str:
    """
    Generate a sanitized and formatted user profile block for web display.

    Sanitizes the given fields to prevent HTML injection and formats them into
    the PROFILE_FORMAT template, inserting safe <br> tags where line breaks
    are present.

    Args:
        username: The user's full name.
        about: A multi-sentence biography provided by the user.
        current_status: A brief, current status update from the user.

    Returns:
        A string containing the sanitized and formatted profile information,
        suitable for direct inclusion in an HTML page.

    Example:
        >>> display_user_profile("Ada Lovelace", "Math pioneer.", "Online")
        'User: Ada Lovelace<br>Bio: Math pioneer.<br>Status: Online'
    """
    # Sanitize each user-provided field for safe HTML rendering.
    safe_username = _sanitize_text(username)
    safe_about = _sanitize_text(about)
    safe_status = _sanitize_text(current_status)

    # Insert sanitized values into the predefined profile template.
    profile_html = PROFILE_FORMAT.format(
        username=safe_username,
        about=safe_about,
        current_status=safe_status,
    )

    # Return the final HTML string ready for display.
    return profile_html
