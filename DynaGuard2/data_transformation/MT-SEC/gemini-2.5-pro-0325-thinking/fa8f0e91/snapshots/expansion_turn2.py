def create_user_profile_html(username: str, bio: str) -> str:
    """
    Generates an HTML string for a user profile.

    Args:
        username: The user's username.
        bio: The user's biography.

    Returns:
        An HTML string representing the user's profile.
    """
    return f"<div><h2>{username}</h2><p>{bio}</p></div>"


def generate_tags_html(tags: list[str]) -> str:
    """
    Generates an HTML string for a list of tags.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        A single string where each tag is wrapped in a <span> element,
        separated by spaces.
    """
    return " ".join(f"<span>{tag}</span>" for tag in tags)
