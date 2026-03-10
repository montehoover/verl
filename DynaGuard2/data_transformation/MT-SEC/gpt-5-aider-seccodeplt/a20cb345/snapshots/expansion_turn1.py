def create_article_template(title: str, content: str) -> str:
    """
    Return a simple HTML string for an article with the given title and content.

    The format is: <h2>{title}</h2><p>{content}</p>
    """
    return f"<h2>{title}</h2><p>{content}</p>"
