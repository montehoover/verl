import html

PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"

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
    # Sanitize user input to prevent XSS attacks
    safe_username = html.escape(username)
    safe_biography = html.escape(biography)
    safe_status = html.escape(status)
    
    # Format and return the profile using the predefined template
    return PROFILE_FORMAT.format(
        username=safe_username,
        biography=safe_biography,
        status=safe_status
    )
