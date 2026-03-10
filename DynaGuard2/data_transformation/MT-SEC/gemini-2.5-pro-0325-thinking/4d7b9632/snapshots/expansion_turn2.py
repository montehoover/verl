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


def generate_tags_html(tags: list[str]) -> str:
    """
    Generates an HTML string for a list of tags.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        A single HTML string with each tag wrapped in a <span> element,
        separated by spaces.
    """
    span_tags = [f"<span>{tag}</span>" for tag in tags]
    return " ".join(span_tags)
