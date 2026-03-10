PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"

def generate_user_profile(username: str, biography: str, status: str) -> str:
    """
    Generates a user profile string formatted for web display.

    Args:
        username: The user's username.
        biography: The user's biography.
        status: The user's current status.

    Returns:
        A formatted string containing the user's profile information.
    """
    return PROFILE_FORMAT.format(username=username, biography=biography, status=status)
