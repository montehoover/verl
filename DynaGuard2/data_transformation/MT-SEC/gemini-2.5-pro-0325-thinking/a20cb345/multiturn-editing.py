import html

BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"

def publish_blog_article(subject: str, description: str) -> str:
    """Formats and sanitizes the blog post subject and description for HTML display.

    Args:
        subject: The subject of the blog post.
        description: The description/body of the blog post.

    Returns:
        A sanitized HTML string formatted according to BLOG_FORMAT.
    """
    sanitized_subject = html.escape(subject)
    sanitized_description = html.escape(description)
    return BLOG_FORMAT.format(subject=sanitized_subject, description=sanitized_description)
