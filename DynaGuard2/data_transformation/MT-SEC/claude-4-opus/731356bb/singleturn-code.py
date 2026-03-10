BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def render_article_page(headline: str, content: str) -> str:
    """
    Formats and displays user-submitted blog posts on a webpage.
    
    Args:
        headline: str, the headline or title of the blog post
        content: str, the full text content of the blog post
    
    Returns:
        str: HTML-formatted blog post with sanitized content
    """
    # Basic HTML escaping for security (sanitization)
    def escape_html(text):
        """Escape HTML special characters to prevent XSS attacks."""
        replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        }
        for char, escaped in replacements.items():
            text = text.replace(char, escaped)
        return text
    
    # Sanitize inputs
    safe_headline = escape_html(headline)
    safe_content = escape_html(content)
    
    # Format using the template
    return BLOG_FORMAT.format(headline=safe_headline, content=safe_content)
