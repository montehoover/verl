import re

def contains_html(text):
    """
    Check if a given string contains any HTML-like content.
    
    Args:
        text (str): The string to check for HTML content
        
    Returns:
        bool: True if HTML-like tags are found, False otherwise
    """
    # Pattern to match HTML tags like <tag> or </tag>
    html_pattern = r'</?[a-zA-Z][^>]*>'
    
    return bool(re.search(html_pattern, text))
