import html

BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"


def sanitize_html(text: str) -> str:
    """
    Escapes special HTML characters in the given text.

    Args:
        text: The string to be sanitized.

    Returns:
        A string with special HTML characters escaped.
    """
    return html.escape(text)


def create_html_paragraph(text: str) -> str:
    """
    Wraps the given text in an HTML <p> element.

    Args:
        text: The string to be wrapped.

    Returns:
        An HTML string with the text wrapped in a <p> element.
    """
    return f"<p>{text}</p>"


def display_blog_entry(headline: str, text: str) -> str:
    """
    Formats and displays a blog post with sanitized inputs.

    Args:
        headline: The headline of the blog post.
        text: The main content of the blog post.

    Returns:
        An HTML string representing the formatted and sanitized blog post.
    """
    sanitized_headline = sanitize_html(headline)
    sanitized_text = sanitize_html(text)
    return BLOG_FORMAT.format(headline=sanitized_headline, text=sanitized_text)
