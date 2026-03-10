import re

def grab_html_tags(html_code: str) -> list:
    """
    Scans a provided string for HTML tags using a regular expression.
    
    Args:
        html_code (str): An HTML string that serves as the input for parsing.
    
    Returns:
        list: A list of all HTML tags identified in the given input.
    """
    # Regular expression pattern to match HTML tags
    # Matches opening tags, closing tags, and self-closing tags
    pattern = r'<[^>]+>'
    
    # Find all matches in the HTML code
    tags = re.findall(pattern, html_code)
    
    return tags
