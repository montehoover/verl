"""
Utilities for generating sanitized HTML snippets for user profiles.

This module exposes a single public function, `display_user_profile`,
which formats user-supplied profile data into a pre-defined HTML template
while ensuring the content is safely escaped for display in a browser.

Logging:
    The module logs profile generation events to aid monitoring and debugging.
    By default, a NullHandler is attached to avoid warnings if the application
    does not configure logging. Applications should configure logging handlers
    and levels as appropriate.
"""

import logging
from html import escape

# Module-level logger for this module.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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

    Logging:
        Logs the start and successful completion of profile generation at INFO
        level, and includes DEBUG details about sanitized content lengths. Any
        exception raised during processing is logged at ERROR level with stack
        trace and re-raised.

    Example:
        >>> display_user_profile("Ada Lovelace", "Math pioneer.", "Online")
        'User: Ada Lovelace<br>Bio: Math pioneer.<br>Status: Online'
    """
    logger.info("Starting profile generation (username=%r)", username)

    try:
        # Sanitize each user-provided field for safe HTML rendering.
        safe_username = _sanitize_text(username)
        safe_about = _sanitize_text(about)
        safe_status = _sanitize_text(current_status)

        logger.debug(
            "Sanitized fields (username=%r, about_len=%d, status_len=%d)",
            safe_username,
            len(safe_about),
            len(safe_status),
        )

        # Insert sanitized values into the predefined profile template.
        profile_html = PROFILE_FORMAT.format(
            username=safe_username,
            about=safe_about,
            current_status=safe_status,
        )

        logger.info(
            "Profile generated successfully (username=%r, html_length=%d)",
            safe_username,
            len(profile_html),
        )

        # Return the final HTML string ready for display.
        return profile_html

    except Exception:
        logger.exception(
            "Failed to generate profile (username=%r)", username
        )
        raise
