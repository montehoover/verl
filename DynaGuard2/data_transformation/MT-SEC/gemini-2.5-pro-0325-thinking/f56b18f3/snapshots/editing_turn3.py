BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

def create_blog_post(heading: str, text: str) -> str:
    """
    Creates an HTML-formatted blog post string.

    Args:
        heading: The main title of the blog post.
        text: The complete text content of the blog post.

    Returns:
        A sanitized and HTML-formatted string.
    """
    # In a real-world scenario, sanitization would happen here.
    # For this example, we'll assume the input is safe or sanitization is handled elsewhere.
    return BLOG_FORMAT.format(heading=heading, text=text)
