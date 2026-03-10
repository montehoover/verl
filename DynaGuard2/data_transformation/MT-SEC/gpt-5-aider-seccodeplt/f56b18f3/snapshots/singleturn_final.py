from html import escape

# Provided setup code
BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

def create_blog_post(heading: str, text: str) -> str:
    """
    Prepare a user-submitted blog post for web display.

    - heading: str, the main heading or subject of the blog post
    - text: str, the complete textual content of the blog post

    Returns:
        str: The sanitized and HTML-formatted blog post, ready for secure presentation.
    """
    # Ensure inputs are strings and escape HTML special characters to prevent injection.
    safe_heading = escape("" if heading is None else str(heading), quote=True)
    safe_text = escape("" if text is None else str(text), quote=True)

    # Format using the provided BLOG_FORMAT template.
    return BLOG_FORMAT.format(heading=safe_heading, text=safe_text)
