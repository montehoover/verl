import html

# Global HTML template for blog post formatting
BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"


def format_blog_html(headline: str, content: str) -> str:
    """
    Formats the blog post using the HTML template.
    
    Args:
        headline: str, the headline or title of the blog post
        content: str, the full text content of the blog post
    
    Returns:
        str: HTML-formatted blog post
    """
    # Apply the HTML template to the provided headline and content
    return BLOG_FORMAT.format(headline=headline, content=content)


def render_article_page(headline: str, content: str) -> str:
    """
    Formats and displays user-submitted blog posts on a webpage.
    
    Args:
        headline: str, the headline or title of the blog post
        content: str, the full text content of the blog post
    
    Returns:
        str: sanitized and HTML-formatted blog post for safe display
    """
    # Sanitize user inputs to prevent XSS attacks
    safe_headline = html.escape(headline)
    safe_content = html.escape(content)
    
    # Format the sanitized content into HTML
    return format_blog_html(safe_headline, safe_content)
