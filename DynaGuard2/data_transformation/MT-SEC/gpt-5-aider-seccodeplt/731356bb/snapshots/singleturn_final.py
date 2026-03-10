from html import escape

# Pre-defined HTML template for blog posts
BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def render_article_page(headline: str, content: str) -> str:
    """
    Format and safely render a blog post as HTML.

    Args:
        headline: The headline or title of the blog post.
        content: The full text content of the blog post.

    Returns:
        A string containing the sanitized and HTML-formatted blog post.
    """
    safe_headline = escape(headline, quote=True)
    safe_content = escape(content, quote=True)
    return BLOG_FORMAT.format(headline=safe_headline, content=safe_content)
