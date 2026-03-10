import html

BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"

def display_blog_entry(headline: str, text: str) -> str:
    """
    Formats and displays user-submitted blog posts on a webpage.

    Args:
        headline: The headline or title of the blog post.
        text: The full text content of the blog post.

    Returns:
        A string with the sanitized and HTML-formatted blog post.
    """
    sanitized_headline = html.escape(headline)
    sanitized_text = html.escape(text)
    return BLOG_FORMAT.format(headline=sanitized_headline, text=sanitized_text)
