def create_user_html(poster_name: str, user_content: str) -> str:
    """
    Generates an HTML string for displaying user information.

    Args:
        poster_name: The name of the poster.
        user_content: The content posted by the user.

    Returns:
        An HTML string with the user's details.
    """
    return f"<div><h2>{poster_name}</h2><p>{user_content}</p></div>"


def generate_tags_html(tags: list[str]) -> str:
    """
    Generates an HTML string for displaying tags.

    Args:
        tags: A list of tags (strings).

    Returns:
        A single string where each tag is enclosed in a <span> element,
        separated by spaces.
    """
    return " ".join(f"<span>{tag}</span>" for tag in tags)
