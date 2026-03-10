def format_user_info(name: str, bio: str) -> str:
    """
    Formats user name and bio into an HTML string.

    Args:
        name: The user's name.
        bio: A short biography of the user.

    Returns:
        An HTML string with the user's information.
    """
    return f"<div><h2>{name}</h2><p>{bio}</p></div>"
