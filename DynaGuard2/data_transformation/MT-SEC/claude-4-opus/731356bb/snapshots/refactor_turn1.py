import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def render_article_page(headline: str, content: str) -> str:
    """
    Formats and displays user-submitted blog posts on a webpage.
    
    Args:
        headline: str, the headline or title of the blog post
        content: str, the full text content of the blog post
    
    Returns:
        str: sanitized and HTML-formatted blog post for safe display
    """
    # Sanitize inputs to prevent XSS attacks
    safe_headline = html.escape(headline)
    safe_content = html.escape(content)
    
    # Format the blog post using the template
    return BLOG_FORMAT.format(headline=safe_headline, content=safe_content)
