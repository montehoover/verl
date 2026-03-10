import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def _sanitize_text(text: str) -> str:
    """Escapes HTML special characters in a string."""
    return html.escape(text)

def format_blog_entry(headline: str, content: str) -> str:
    """
    Prepares user-submitted blog posts for web display.

    Args:
        headline: The main heading or subject of the blog post.
        content: The complete textual content of the blog post.

    Returns:
        A string containing the sanitized and HTML-formatted blog post,
        ready for secure presentation on the webpage.
    """
    sanitized_headline = _sanitize_text(headline)
    sanitized_content = _sanitize_text(content)
    return BLOG_FORMAT.format(headline=sanitized_headline, content=sanitized_content)
