from html import escape

# Use provided BLOG_FORMAT if available; otherwise, define a safe default.
try:
    BLOG_FORMAT  # type: ignore[name-defined]
except NameError:
    BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"


def display_blog_entry(headline: str, text: str) -> str:
    """
    Format and sanitize a blog post for safe HTML display.

    Args:
        headline: the headline or title of the blog post
        text: the full text content of the blog post

    Returns:
        A string containing sanitized HTML formatted using BLOG_FORMAT.
    """
    # Coerce to strings and handle None safely
    headline_str = "" if headline is None else str(headline)
    text_str = "" if text is None else str(text)

    # Escape HTML special characters to prevent injection
    safe_headline = escape(headline_str, quote=True)
    safe_text = escape(text_str, quote=True)

    # Preserve line breaks safely by converting newlines to <br>
    safe_text = safe_text.replace("\r\n", "\n").replace("\r", "\n")
    safe_text = safe_text.replace("\n", "<br>")

    return BLOG_FORMAT.format(headline=safe_headline, text=safe_text)
