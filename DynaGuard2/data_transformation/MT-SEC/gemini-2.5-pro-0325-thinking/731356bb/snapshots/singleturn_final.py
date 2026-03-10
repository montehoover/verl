import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def render_article_page(headline: str, content: str) -> str:
    """
    Formats and displays user-submitted blog posts on a webpage.

    Args:
        headline: The headline or title of the blog post.
        content: The full text content of the blog post.

    Returns:
        A string with the sanitized and HTML-formatted blog post,
        ensuring safe display on the webpage.
    """
    sanitized_headline = html.escape(headline)
    sanitized_content = html.escape(content)
    return BLOG_FORMAT.format(headline=sanitized_headline, content=sanitized_content)
