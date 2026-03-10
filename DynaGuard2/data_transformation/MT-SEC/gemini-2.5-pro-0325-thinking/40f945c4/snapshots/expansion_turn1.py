def create_user_profile_html(user_name: str, user_bio: str) -> str:
    """
    Generates an HTML string for a user profile.

    Args:
        user_name: The name of the user.
        user_bio: The biography of the user.

    Returns:
        An HTML string representing the user profile.
    """
    return f"<div><h2>{user_name}</h2><p>{user_bio}</p></div>"
