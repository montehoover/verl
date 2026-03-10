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

def extract_html_attributes(tag):
    """
    Extract attributes from an HTML tag.
    
    Args:
        tag (str): The HTML tag string to parse
        
    Returns:
        dict: Dictionary of attribute names and values
    """
    attributes = {}
    
    # Remove the < and > characters and get the tag content
    tag_content = tag.strip('<>/')
    
    # Split tag name from attributes
    parts = tag_content.split(None, 1)
    if len(parts) < 2:
        return attributes
    
    attributes_string = parts[1]
    
    # Pattern to match attributes (handles both single and double quotes)
    attr_pattern = r'(\w+)=(["\'])([^"\']*)\2'
    
    # Find all attribute matches
    matches = re.findall(attr_pattern, attributes_string)
    
    for match in matches:
        attr_name = match[0]
        attr_value = match[2]
        attributes[attr_name] = attr_value
    
    return attributes

def parse_html_tags(html):
    """
    Extract all HTML tags from a given string.
    
    Args:
        html (str): The HTML string to parse
        
    Returns:
        list: List of all HTML tags found in the input string
    """
    # Pattern to match HTML tags (opening, closing, or self-closing)
    tag_pattern = r'<[^>]+>'
    
    # Find all HTML tags in the string
    tags = re.findall(tag_pattern, html)
    
    return tags
