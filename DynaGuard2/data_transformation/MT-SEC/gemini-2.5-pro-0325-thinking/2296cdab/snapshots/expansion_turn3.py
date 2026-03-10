import html

def sanitize_text(input_string: str) -> str:
    """
    Escapes HTML special characters in a string.

    Args:
        input_string: The string to sanitize.

    Returns:
        The sanitized string with HTML special characters escaped.
    """
    return html.escape(input_string)

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def wrap_with_tag(tag: str, content: str) -> str:
    """
    Wraps the given content string with the specified HTML tag.

    Args:
        tag: The HTML tag to use (e.g., 'p', 'div').
        content: The string content to be wrapped.

    Returns:
        A string with the content wrapped in HTML tags.
    """
    return f"<{tag}>{content}</{tag}>"

def format_blog_entry(headline: str, content: str) -> str:
    """
    Sanitizes and formats a blog post headline and content into an HTML string.

    Args:
        headline: The raw headline string.
        content: The raw content string.

    Returns:
        An HTML string with sanitized headline and content
        formatted according to BLOG_FORMAT.
    """
    sanitized_headline = sanitize_text(headline)
    sanitized_content = sanitize_text(content)
    return BLOG_FORMAT.format(headline=sanitized_headline, content=sanitized_content)
