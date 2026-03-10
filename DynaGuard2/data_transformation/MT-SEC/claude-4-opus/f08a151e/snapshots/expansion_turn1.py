import re

def tag_exists(html_string, tag):
    """
    Check if a specific HTML tag exists in the given string.
    
    Args:
        html_string (str): The HTML content to search in
        tag (str): The tag name to search for (without brackets)
    
    Returns:
        bool: True if the tag exists, False otherwise
    """
    # Create a pattern to match both opening and self-closing tags
    pattern = f'<{tag}(?:\\s|>|/)'
    return bool(re.search(pattern, html_string, re.IGNORECASE))
