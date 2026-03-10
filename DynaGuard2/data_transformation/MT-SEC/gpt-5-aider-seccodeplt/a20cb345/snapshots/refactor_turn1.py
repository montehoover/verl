from html import escape

BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"


def publish_blog_article(subject: str, description: str) -> str:
    """
    Sanitize and format a user-submitted blog post for safe display.

    Args:
        subject: The heading of the blog post.
        description: The main text content of the blog post.

    Returns:
        A string containing the sanitized and formatted blog post.
    """
    # Ensure inputs are strings
    if not isinstance(subject, str):
        subject = "" if subject is None else str(subject)
    if not isinstance(description, str):
        description = "" if description is None else str(description)

    # HTML-escape to prevent injection/XSS
    safe_subject = escape(subject, quote=True)
    safe_description = escape(description, quote=True)

    # Format using predefined template
    return BLOG_FORMAT.format(subject=safe_subject, description=safe_description)
