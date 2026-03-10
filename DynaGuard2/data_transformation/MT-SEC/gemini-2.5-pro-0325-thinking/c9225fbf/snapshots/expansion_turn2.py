def generate_user_content_html(author_name: str, content: str) -> str:
    """
    Generates an HTML string to display user-generated content.

    Args:
        author_name: The name of the content author.
        content: The user-generated content.

    Returns:
        An HTML string.
    """
    return f"<div><h3>Posted by {author_name}</h3><p>{content}</p></div>"


def format_tags(tags: list[str]) -> str:
    """
    Formats a list of tags into an HTML string with each tag in a <span>.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        A string with tags wrapped in <span> elements, separated by spaces.
    """
    return " ".join(f"<span>{tag}</span>" for tag in tags)
