import html

# Pre-defined HTML template for blog posts
BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"


def display_blog_entry(headline: str, text: str) -> str:
    """
    Format and sanitize a user-submitted blog post for safe HTML display.

    Args:
        headline (str): The headline or title of the blog post.
        text (str): The full text content of the blog post.

    Returns:
        str: The sanitized and HTML-formatted blog post.
    """
    # Coerce to strings and handle None safely
    if headline is None:
        headline = ""
    if text is None:
        text = ""

    # Escape HTML special characters to prevent XSS
    safe_headline = html.escape(str(headline), quote=True)
    safe_text = html.escape(str(text), quote=True)

    # Format using the provided template
    return BLOG_FORMAT.format(headline=safe_headline, text=safe_text)
