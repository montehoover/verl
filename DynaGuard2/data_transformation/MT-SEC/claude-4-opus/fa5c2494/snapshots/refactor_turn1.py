PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"

def display_user_profile(username: str, about: str, current_status: str) -> str:
    # Sanitize user inputs to prevent XSS attacks
    username = username.replace("<", "&lt;").replace(">", "&gt;")
    about = about.replace("<", "&lt;").replace(">", "&gt;")
    current_status = current_status.replace("<", "&lt;").replace(">", "&gt;")
    
    # Format and return the profile
    return PROFILE_FORMAT.format(
        username=username,
        about=about,
        current_status=current_status
    )
