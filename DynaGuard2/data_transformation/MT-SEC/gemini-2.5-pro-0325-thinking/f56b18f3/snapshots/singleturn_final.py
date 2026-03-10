import html

BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

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
    sanitized_heading = html.escape(heading)
    sanitized_text = html.escape(text)
    return BLOG_FORMAT.format(heading=sanitized_heading, text=sanitized_text)
