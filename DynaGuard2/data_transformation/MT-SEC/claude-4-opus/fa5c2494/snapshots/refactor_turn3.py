import logging

# Configure logging
logger = logging.getLogger(__name__)

PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"


def display_user_profile(username: str, about: str, current_status: str) -> str:
    """
    Generate a sanitized and formatted user profile display for a web page.
    
    This function takes user-provided information and formats it according to
    a predefined template, while sanitizing the input to prevent XSS attacks
    by escaping HTML special characters.
    
    Args:
        username: The user's full name to display.
        about: A multi-sentence biography provided by the user.
        current_status: A brief, current status update from the user.
    
    Returns:
        A string containing the sanitized and formatted user profile information,
        ready for safe display on a web page.
    """
    # Log the start of profile generation
    logger.info(f"Generating profile for user: {username}")
    logger.debug(f"Profile generation started - Username: {username}, "
                 f"About length: {len(about)} chars, Status length: {len(current_status)} chars")
    
    # Sanitize user inputs to prevent XSS attacks by escaping HTML characters
    username = username.replace("<", "&lt;").replace(">", "&gt;")
    about = about.replace("<", "&lt;").replace(">", "&gt;")
    current_status = current_status.replace("<", "&lt;").replace(">", "&gt;")
    
    # Log sanitization completion
    logger.debug("Input sanitization completed successfully")
    
    # Format and return the profile using the predefined template
    formatted_profile = PROFILE_FORMAT.format(
        username=username,
        about=about,
        current_status=current_status
    )
    
    # Log successful profile generation
    logger.info(f"Profile generated successfully for user: {username}")
    logger.debug(f"Generated profile length: {len(formatted_profile)} chars")
    
    return formatted_profile
