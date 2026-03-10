import re

def contains_html_tags(text):
    """
    Check if a given string contains any HTML tags.
    
    Args:
        text (str): The string to check for HTML tags
        
    Returns:
        bool: True if HTML tags are present, False otherwise
    """
    try:
        # Regular expression pattern to match HTML tags
        # Matches opening tags like <tag>, closing tags like </tag>, and self-closing tags like <tag/>
        html_pattern = r'<[^>]+>'
        
        # Search for HTML tags in the text
        match = re.search(html_pattern, text)
        
        # Return True if match found, False otherwise
        return match is not None
    except:
        # Return False if any exception occurs
        return False
