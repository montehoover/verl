BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def format_blog_entry(headline, content):
    """
    Format blog entry for safe storage with HTML formatting.
    
    Args:
        headline (str): The main heading of the blog post
        content (str): The complete textual content of the blog post
        
    Returns:
        str: Sanitized HTML formatted string
    """
    if not isinstance(headline, str):
        headline = str(headline)
    if not isinstance(content, str):
        content = str(content)
    
    # HTML escape special characters
    html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&#x27;",
        ">": "&gt;",
        "<": "&lt;",
    }
    
    # Escape headline
    for char, escape in html_escape_table.items():
        headline = headline.replace(char, escape)
    
    # Escape content
    for char, escape in html_escape_table.items():
        content = content.replace(char, escape)
    
    # Remove any null bytes
    headline = headline.replace('\x00', '')
    content = content.replace('\x00', '')
    
    # Normalize whitespace
    headline = ' '.join(headline.split())
    content = ' '.join(content.split())
    
    # Format using the predefined template
    formatted_html = BLOG_FORMAT.format(headline=headline, content=content)
    
    return formatted_html
