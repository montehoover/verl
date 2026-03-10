import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='profile_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"

def sanitize_html(text: str) -> str:
    """
    Sanitizes text to prevent XSS attacks by escaping HTML special characters.
    
    Args:
        text: The text to sanitize
    
    Returns:
        The sanitized text with HTML special characters escaped
    """
    return text.replace('<', '&lt;').replace('>', '&gt;')

def format_profile(username: str, biography: str, status: str) -> str:
    """
    Formats user profile data using the PROFILE_FORMAT template.
    
    Args:
        username: The user's name
        biography: The user's biography
        status: The user's current status
    
    Returns:
        The formatted profile string
    """
    return PROFILE_FORMAT.format(
        username=username,
        biography=biography,
        status=status
    )

def generate_user_profile(username: str, biography: str, status: str) -> str:
    """
    Renders a user's profile information for web display.
    
    Args:
        username: The user's name as they've entered it
        biography: A user-written biography that may span multiple sentences
        status: A brief, current status message provided by the user
    
    Returns:
        A formatted string containing the sanitized user profile information ready for web display
    """
    # Log the profile generation activity
    logging.info(f"Profile generated for user: {username}")
    
    # Sanitize user input to prevent XSS attacks
    sanitized_username = sanitize_html(username)
    sanitized_biography = sanitize_html(biography)
    sanitized_status = sanitize_html(status)
    
    # Format and return the profile
    return format_profile(sanitized_username, sanitized_biography, sanitized_status)
