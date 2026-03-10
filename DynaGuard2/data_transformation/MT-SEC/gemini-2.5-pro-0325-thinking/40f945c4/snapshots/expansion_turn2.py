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


def generate_tags_html(tags: list[str]) -> str:
    """
    Generates an HTML string for a list of tags.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        A single string where each tag is enclosed in a <span> element,
        separated by spaces.
    """
    return " ".join(f"<span>{tag}</span>" for tag in tags)
