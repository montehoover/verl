import html
import logging
import datetime

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create a file handler
handler = logging.FileHandler('profile_generation.log')
handler.setLevel(logging.INFO)
# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(handler)


PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"

def _sanitize_input(text: str) -> str:
    """Sanitizes a string for safe HTML display."""
    return html.escape(text)

def _format_profile_data(username: str, biography: str, status: str) -> str:
    """Formats the profile data using the predefined template."""
    return PROFILE_FORMAT.format(username=username, biography=biography, status=status)

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
    logger.info(f"Generating profile for user: {username}")
    # For web display, it's crucial to sanitize inputs to prevent XSS.
    sanitized_username = _sanitize_input(username)
    sanitized_biography = _sanitize_input(biography)
    sanitized_status = _sanitize_input(status)

    return _format_profile_data(
        username=sanitized_username,
        biography=sanitized_biography,
        status=sanitized_status
    )
