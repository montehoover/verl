def generate_post_html(title: str, content: str) -> str:
    """
    Generates an HTML representation of a forum post.

    Args:
        title: The title of the post.
        content: The content of the post.

    Returns:
        An HTML string representing the post.
    """
    return f"<div><h2>{title}</h2><p>{content}</p></div>"
