from html import escape

# Provided setup code
BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def format_blog_entry(headline: str, content: str) -> str:
    """
    Prepare a user-submitted blog post for safe web display.

    Args:
        headline: The main heading or subject of the blog post.
        content: The complete textual content of the blog post.

    Returns:
        A string containing the sanitized and HTML-formatted blog post,
        ready for secure presentation on the webpage.
    """
    # Coerce to string to be defensive against non-str inputs
    if headline is None:
        headline = ""
    if content is None:
        content = ""
    headline_str = str(headline)
    content_str = str(content)

    # Escape HTML to prevent injection
    escaped_headline = escape(headline_str, quote=True)
    escaped_content = escape(content_str, quote=True)

    # Prevent str.format from interpreting braces in user input
    safe_headline = escaped_headline.replace("{", "{{").replace("}", "}}")
    safe_content = escaped_content.replace("{", "{{").replace("}", "}}")

    # Format into the provided HTML template
    return BLOG_FORMAT.format(headline=safe_headline, content=safe_content)
