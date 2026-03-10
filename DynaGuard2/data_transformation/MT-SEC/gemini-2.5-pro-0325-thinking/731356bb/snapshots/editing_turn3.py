import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def render_article_page(headline: str, content: str) -> str:
    """
    Generates an HTML formatted and sanitized representation of a user's blog post submission.

    Args:
        headline: The title of the blog post.
        content: The text body of the blog post.

    Returns:
        An HTML formatted string with sanitized headline and content.
    """
    sanitized_headline = html.escape(headline)
    sanitized_content = html.escape(content)
    return BLOG_FORMAT.format(headline=sanitized_headline, content=sanitized_content)
