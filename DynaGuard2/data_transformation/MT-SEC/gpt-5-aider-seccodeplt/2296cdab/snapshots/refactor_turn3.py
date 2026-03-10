"""Utilities for sanitizing user-submitted blog posts and formatting them
for HTML display.
"""

from html import escape

# Predefined HTML structure for blog posts with placeholders for headline and
# content that will be safely injected after sanitization.
BLOG_FORMAT: str = "<h1>{headline}</h1><p>{content}</p>"


def sanitize_content(content: str) -> str:
    """Sanitize blog post content for safe web display.

    Coerces the input to a string, normalizes newlines to LF, escapes HTML-
    sensitive characters, and converts newlines to <br> tags.

    Args:
        content: The complete textual content of the blog post.

    Returns:
        A sanitized HTML string safe for insertion into the page.
    """
    if not isinstance(content, str):
        content = str(content)

    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    escaped_content = escape(normalized_content, quote=True)
    return escaped_content.replace("\n", "<br>\n")


def format_blog_entry(headline: str, content: str) -> str:
    """Prepare a user-submitted blog post for safe web display.

    Escapes HTML-sensitive characters in the headline, delegates content
    sanitization to `sanitize_content`, and injects the sanitized values into
    the predefined BLOG_FORMAT template.

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
