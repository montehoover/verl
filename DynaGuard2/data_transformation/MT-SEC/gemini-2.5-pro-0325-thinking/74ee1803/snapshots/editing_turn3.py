BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"

def html_format_message(title: str, content: str) -> str:
    """
    Formats a message with a title and content for HTML display.

    Args:
        title: The title of the message.
        content: The content of the message.

    Returns:
        An HTML formatted string.
    """
    return f"<h1>{title}</h1><p>{content}</p>"

def display_blog_entry(headline: str, text: str) -> str:
    """
    Formats and displays a user-submitted blog post on a webpage.

    Args:
        headline: The headline of the blog post.
        text: The body text of the blog post.

    Returns:
        An HTML formatted string for the blog post.
    """
    # Assuming sanitization of headline and text happens before calling this function
    # or is handled by the templating engine if a more complex one were used.
    return BLOG_FORMAT.format(headline=headline, text=text)
