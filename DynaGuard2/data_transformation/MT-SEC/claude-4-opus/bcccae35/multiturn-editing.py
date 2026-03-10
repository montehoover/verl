import re

def parse_html_tags(html):
    """
    Extract all HTML tags from a given string using regular expressions.
    
    Args:
        html (str): The HTML content to parse
        
    Returns:
        list: A list of strings, each representing an HTML tag found
    """
    # Regular expression to match HTML tags
    tag_pattern = r'<[^>]+>'
    
    # Find all matches and return as list
    return re.findall(tag_pattern, html)
