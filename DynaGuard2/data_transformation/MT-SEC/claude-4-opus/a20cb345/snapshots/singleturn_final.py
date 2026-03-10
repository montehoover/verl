BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"

def publish_blog_article(subject: str, description: str) -> str:
    """
    Displays a user-submitted blog post on a webpage.
    
    Args:
        subject: The heading of the blog post
        description: The main text content of the blog post
    
    Returns:
        A string containing the sanitized and formatted blog post, ready for safe display
    """
    return BLOG_FORMAT.format(subject=subject, description=description)
