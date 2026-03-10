"""Utilities for sanitizing and rendering user-submitted blog posts into HTML."""

from html import escape


# Default fallback used if a global BLOG_FORMAT template is not provided.
DEFAULT_BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"


def format_blog_html(template: str, headline: str, content: str) -> str:
    """
    Format a blog post into HTML using the provided template.

    This is a pure function: given a template and sanitized inputs, it returns a
    deterministic HTML string without side effects.

    Args:
        template: HTML template with {headline} and {content} placeholders.
        headline: Sanitized headline text to insert.
        content: Sanitized content text to insert.

    Returns:
        The HTML-formatted blog post string.
    """
    # Interpolate the sanitized values into the template placeholders.
    return template.format(headline=headline, content=content)


def render_article_page(headline: str, content: str) -> str:
    """
    Sanitize and format a blog post into HTML using a template.

    Uses the global BLOG_FORMAT template when available; otherwise, falls back
    to DEFAULT_BLOG_FORMAT.

    Args:
        headline: The headline or title of the blog post.
        content: The full text content of the blog post.

    Returns:
        A string with the sanitized and HTML-formatted blog post.
    """
    # Sanitize user-provided inputs to prevent HTML injection and XSS.
    safe_headline = escape(headline, quote=True)
    safe_content = escape(content, quote=True)

    # Prefer a globally provided template named BLOG_FORMAT if present.
    template = globals().get("BLOG_FORMAT", DEFAULT_BLOG_FORMAT)

    # Delegate to the pure formatter with sanitized values.
    return format_blog_html(template, safe_headline, safe_content)
