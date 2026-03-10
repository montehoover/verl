def create_post_html(headline: str, content: str) -> str:
    """
    Create an HTML snippet for a forum post.

    Args:
        headline: The post headline.
        content: The post content.

    Returns:
        An HTML string in the format: <h2>{headline}</h2><p>{content}</p>
    """
    return f"<h2>{headline}</h2><p>{content}</p>"
