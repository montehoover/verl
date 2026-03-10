def create_post_html(headline: str, content: str) -> str:
    """
    Generates an HTML string for a forum post.

    Args:
        headline: The headline of the post.
        content: The main content of the post.

    Returns:
        An HTML string formatted as a header and a paragraph.
    """
    return f"<h2>{headline}</h2><p>{content}</p>"
