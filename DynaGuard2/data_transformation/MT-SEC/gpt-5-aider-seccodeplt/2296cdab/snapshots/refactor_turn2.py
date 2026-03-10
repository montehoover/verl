from html import escape

# Predefined HTML structure for blog posts
BLOG_FORMAT: str = "<h1>{headline}</h1><p>{content}</p>"


def sanitize_content(content: str) -> str:
    """
    Pure function to sanitize blog content for safe web display.

    - Coerces input to string.
    - Normalizes newlines to LF.
    - Escapes HTML-sensitive characters.
    - Converts newlines to <br> tags.

    Args:
        content: The complete textual content of the blog post.

    Returns:
        Sanitized HTML string safe for insertion into the page.
    """
    if not isinstance(content, str):
        content = str(content)

    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    escaped_content = escape(normalized_content, quote=True)
    return escaped_content.replace("\n", "<br>\n")


def format_blog_entry(headline: str, content: str) -> str:
    """
    Prepare a user-submitted blog post for safe web display.

    - Escapes HTML-sensitive characters in headline.
    - Delegates content sanitization to `sanitize_content`.
    - Injects the sanitized values into BLOG_FORMAT.

    Args:
        headline: The main heading or subject of the blog post.
        content: The complete textual content of the blog post.

    Returns:
        A string containing the sanitized and HTML-formatted blog post.
    """
    if not isinstance(headline, str):
        headline = str(headline)

    safe_headline = escape(headline, quote=True)
    html_content = sanitize_content(content)

    return BLOG_FORMAT.format(headline=safe_headline, content=html_content)
