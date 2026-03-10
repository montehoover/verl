import re

def parse_html_tags(html: str):
    """
    Extract all HTML tags from a given string using regular expressions.
    
    Args:
        html: str - The input HTML string to be parsed
        
    Returns:
        List of strings, each representing an HTML tag found in the input
    """
    # Regular expression to match HTML tags
    # This pattern matches both opening and closing tags with optional attributes
    pattern = r'<([^>]+)>'
    
    # Find all matches
    matches = re.findall(pattern, html)
    
    return matches
