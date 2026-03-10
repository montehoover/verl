from html import escape

def render_article_page(headline: str, content: str) -> str:
    """
    Sanitize and format a blog post into HTML using the global BLOG_FORMAT template.

    Args:
        headline: the headline or title of the blog post.
        content: the full text content of the blog post.

    Returns:
        A string with the sanitized and HTML-formatted blog post.
    """
    safe_headline = escape(headline, quote=True)
    safe_content = escape(content, quote=True)

    try:
        template = BLOG_FORMAT  # type: ignore[name-defined]
    except NameError:
        template = "<h1>{headline}</h1><p>{content}</p>"

    return template.format(headline=safe_headline, content=safe_content)
