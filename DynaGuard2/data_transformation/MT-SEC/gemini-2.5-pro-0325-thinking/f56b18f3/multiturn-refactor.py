import html

BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"


def _sanitize_input(text: str) -> str:
    """Sanitizes a string for safe HTML display."""
    # Guard clause for input type: ensures 'text' is a string.
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string to be sanitized.")
    return html.escape(text)


def _format_as_html(heading: str, text: str) -> str:
    """Formats the heading and text into an HTML string."""
    # Guard clauses for input types: ensure 'heading' and 'text' are strings.
    if not isinstance(heading, str):
        raise TypeError("Input 'heading' must be a string for HTML formatting.")
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string for HTML formatting.")
    return BLOG_FORMAT.format(heading=heading, text=text)


def create_blog_post(heading: str, text: str) -> str:
    """
    Prepares user-submitted blog posts for web display.

    Args:
        heading: The main heading or subject of the blog post.
        text: The complete textual content of the blog post.

    Returns:
        A string containing the sanitized and HTML-formatted blog post,
        ready for secure presentation on the webpage.
    """
    sanitized_heading = _sanitize_input(heading)
    sanitized_text = _sanitize_input(text)
    return _format_as_html(sanitized_heading, sanitized_text)
