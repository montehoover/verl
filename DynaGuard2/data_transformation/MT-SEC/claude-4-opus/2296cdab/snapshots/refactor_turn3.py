import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"


def sanitize_text(text: str) -> str:
    """Sanitize text by escaping HTML special characters.
    
    Args:
        text: The text to be sanitized.
        
    Returns:
        The sanitized text with HTML special characters escaped.
    """
    return html.escape(text)


def format_blog_entry(headline: str, content: str) -> str:
    """Format a blog entry with sanitized headline and content.
    
    Args:
        headline: The main heading or subject of the blog post.
        content: The complete textual content of the blog post.
        
    Returns:
        A string containing the sanitized and HTML-formatted blog post,
        ready for secure presentation on the webpage.
    """
    sanitized_headline = sanitize_text(headline)
    sanitized_content = sanitize_text(content)
    return BLOG_FORMAT.format(headline=sanitized_headline, content=sanitized_content)
