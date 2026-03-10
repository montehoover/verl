BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"

def publish_blog_article(subject: str, description: str) -> str:
    """
    Formats a blog post for display on a webpage.

    Args:
        subject: The heading of the blog post.
        description: The main text content of the blog post.

    Returns:
        A string containing the formatted blog post, ready for display.
    """
    # For now, we are directly formatting. If sanitization is needed,
    # a library like bleach or markupsafe should be used.
    return BLOG_FORMAT.format(subject=subject, description=description)
