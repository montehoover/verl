import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def format_blog_entry(headline: str, content: str) -> str:
    """
    Prepares user-submitted blog posts for web display.
    
    Args:
        headline: str, the main heading or subject of the blog post
        content: str, the complete textual content of the blog post
    
    Returns:
        str: a string containing the sanitized and HTML-formatted blog post,
             ready for secure presentation on the webpage
    """
    # Sanitize inputs by escaping HTML special characters
    sanitized_headline = html.escape(headline)
    sanitized_content = html.escape(content)
    
    # Format using the provided template
    return BLOG_FORMAT.format(headline=sanitized_headline, content=sanitized_content)
