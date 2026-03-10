import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def _format_blog_post_html(headline: str, content: str) -> str:
    """
    Formats the blog post headline and content into an HTML string.

    Args:
        headline: The sanitized headline of the blog post.
        content: The sanitized content of the blog post.

    Returns:
        An HTML string representing the blog post.
    """
    return BLOG_FORMAT.format(headline=headline, content=content)

def render_article_page(headline: str, content: str) -> str:
    """
    Formats and displays user-submitted blog posts on a webpage.

    Args:
        headline: The headline or title of the blog post.
        content: The full text content of the blog post.

    Returns:
        A string with the sanitized and HTML-formatted blog post.
    """
    sanitized_headline = html.escape(headline)
    sanitized_content = html.escape(content)
    return _format_blog_post_html(headline=sanitized_headline, content=sanitized_content)
