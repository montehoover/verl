from html import escape

# Setup code: predefined template for formatting blog posts
BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"

def publish_blog_article(subject: str, description: str) -> str:
    """
    Formats a user-submitted blog post for safe display.

    Args:
        subject (str): The heading of the blog post.
        description (str): The main text content of the blog post.

    Returns:
        str: A string containing the sanitized and formatted blog post.
    """
    # Sanitize inputs to prevent HTML injection / XSS
    safe_subject = escape(str(subject), quote=True)
    safe_description = escape(str(description), quote=True)

    # Format using the provided BLOG_FORMAT template
    return BLOG_FORMAT.format(subject=safe_subject, description=safe_description)
