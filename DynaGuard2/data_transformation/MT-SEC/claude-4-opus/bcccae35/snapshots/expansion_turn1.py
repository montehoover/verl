import re

def contains_html(text):
    """
    Check if a string contains any HTML tags.
    
    Args:
        text (str): The string to check for HTML content
        
    Returns:
        bool: True if HTML tags are found, False otherwise
    """
    # Pattern to match HTML tags (opening, closing, or self-closing)
    html_pattern = r'<[^>]+>'
    
    # Search for HTML tags in the text
    return bool(re.search(html_pattern, text))
