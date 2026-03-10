def create_user_html(author_name: str, user_bio: str) -> str:
    """
    Generates an HTML string displaying user information.

    Args:
        author_name: The name of the user.
        user_bio: A short biography of the user.

    Returns:
        An HTML string with the user's information.
    """
    return f"<div><h2>{author_name}</h2><p>{user_bio}</p></div>"
