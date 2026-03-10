def create_user_profile_html(username: str, bio: str) -> str:
    """
    Generates an HTML string for a user profile.

    Args:
        username: The user's username.
        bio: The user's biography.

    Returns:
        An HTML string representing the user profile.
    """
    return f"<div><h2>{username}</h2><p>{bio}</p></div>"
