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


def extract_tag_attributes(tag):
    """
    Extract attributes from a single HTML tag.
    
    Args:
        tag (str): A string representing a single HTML tag
        
    Returns:
        dict: Dictionary with attribute names as keys and attribute values as values
    """
    attributes = {}
    
    # Pattern to match attribute="value" or attribute='value' or attribute=value
    attr_pattern = r'(\w+)\s*=\s*["\']?([^"\'>\s]+)["\']?'
    
    matches = re.findall(attr_pattern, tag)
    
    for attr_name, attr_value in matches:
        attributes[attr_name] = attr_value
    
    return attributes


def detect_html_tags(html_str):
    """
    Find and return all HTML tags in an input string.
    
    Args:
        html_str (str): The string to parse for HTML tags
        
    Returns:
        list: A list of strings representing the tags discovered
    """
    try:
        # Pattern to match complete HTML tags including opening, closing, and self-closing
        tag_pattern = r'<[^>]+>'
        
        tags = re.findall(tag_pattern, html_str)
        
        return tags
    except:
        return []
