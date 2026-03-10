import os
import logging
from html import escape

PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"

# Logging setup: human-readable log lines to a file in the current working directory
LOG_FILE = os.path.join(os.getcwd(), "profile_generation.log")
_logger = logging.getLogger("profile_generation")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _formatter = logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    _logger.propagate = False


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


def _safe_username_for_log(username: str) -> str:
    if not isinstance(username, str):
        return "unknown"
    cleaned = username.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
    return cleaned or "unknown"


def _log_profile_generation(username: str) -> None:
    try:
        uname = _safe_username_for_log(username)
        _logger.info("Generated profile for user '%s'", uname)
    except Exception:
        # Do not let logging failures impact profile generation.
        pass


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
    result = format_profile(sanitized)
    _log_profile_generation(username)
    return result
