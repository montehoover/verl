from html import escape

# Predefined HTML structure for blog posts.
BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

def create_blog_post(heading: str, text: str) -> str:
    """
    Prepare a user-submitted blog post for safe HTML display.

    - Escapes HTML in heading and text to prevent XSS.
    - Normalizes line endings and converts newlines in text to <br> tags.
    - Formats the output using BLOG_FORMAT.
    """
    if not isinstance(heading, str):
        raise TypeError("heading must be a str")
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    # Normalize whitespace and line endings
    normalized_heading = heading.strip()
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Escape HTML and convert newlines to <br> for display
    safe_heading = escape(normalized_heading, quote=True)
    safe_text = escape(normalized_text, quote=True).replace("\n", "<br>")

    return BLOG_FORMAT.format(heading=safe_heading, text=safe_text)
