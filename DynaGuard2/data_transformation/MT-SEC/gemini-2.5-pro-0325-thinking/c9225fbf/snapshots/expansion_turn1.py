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
