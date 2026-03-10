def create_article_template(title: str, content: str) -> str:
    """
    Creates a simple HTML string for an article.

    Args:
        title: The title of the article.
        content: The content of the article.

    Returns:
        An HTML string with the title in <h2> tags and content in <p> tags.
    """
    return f"<h2>{title}</h2><p>{content}</p>"
