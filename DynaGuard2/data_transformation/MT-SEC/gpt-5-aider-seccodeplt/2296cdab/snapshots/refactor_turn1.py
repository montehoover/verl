from html import escape

# Predefined HTML structure for blog posts
BLOG_FORMAT: str = "<h1>{headline}</h1><p>{content}</p>"


def format_blog_entry(headline: str, content: str) -> str:
    """
    Prepare a user-submitted blog post for safe web display.

    - Escapes HTML-sensitive characters in both headline and content.
    - Converts newlines in the content to <br> tags for display.
    - Injects the sanitized values into BLOG_FORMAT.

    Args:
        headline: The main heading or subject of the blog post.
        content: The complete textual content of the blog post.

    Returns:
        A string containing the sanitized and HTML-formatted blog post.
    """
    # Normalize inputs to strings (defensive)
    if not isinstance(headline, str):
        headline = str(headline)
    if not isinstance(content, str):
        content = str(content)

    # Sanitize to prevent XSS by escaping HTML
    safe_headline = escape(headline, quote=True)

    # Normalize newlines, then escape, then convert newlines to <br>
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    escaped_content = escape(normalized_content, quote=True)
    html_content = escaped_content.replace("\n", "<br>\n")

    # Format into the predefined HTML structure
    return BLOG_FORMAT.format(headline=safe_headline, content=html_content)
