import html


BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"  # HTML template for blog posts


def display_blog_entry(headline: str, text: str) -> str:
    """
    Formats and displays user-submitted blog posts on a webpage.

    The function first sanitizes the input headline and text to prevent
    cross-site scripting (XSS) vulnerabilities by escaping HTML special
    characters. Then, it formats the sanitized content using a predefined
    HTML template.

    Args:
        headline: The headline or title of the blog post.
        text: The full text content of the blog post.

    Returns:
        A string with the sanitized and HTML-formatted blog post,
        safe for display on a webpage.
    """
    # Sanitize inputs to prevent XSS by escaping special HTML characters.
    # This is crucial for security to avoid rendering malicious HTML/script content.
    sanitized_headline = html.escape(headline)
    sanitized_text = html.escape(text)

    # Format the sanitized content into the blog post HTML structure.
    return BLOG_FORMAT.format(headline=sanitized_headline, text=sanitized_text)
