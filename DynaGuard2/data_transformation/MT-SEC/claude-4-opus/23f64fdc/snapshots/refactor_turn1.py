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
    username = username.replace('<', '&lt;').replace('>', '&gt;')
    biography = biography.replace('<', '&lt;').replace('>', '&gt;')
    status = status.replace('<', '&lt;').replace('>', '&gt;')
    
    # Format and return the profile
    return PROFILE_FORMAT.format(
        username=username,
        biography=biography,
        status=status
    )
