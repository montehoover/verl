import re

def identify_html_tags(content: str):
    """
    Scans a provided string for HTML tags using a regular expression.
    
    Args:
        content: str - an HTML string that serves as the input for parsing
        
    Returns:
        Returns a list of all HTML tags identified in the given input
    """
    # Regular expression pattern to match HTML tags
    # This pattern matches opening tags, closing tags, and self-closing tags
    tag_pattern = r'<[^>]+>'
    
    # Find all matches of the pattern in the content
    tags = re.findall(tag_pattern, content)
    
    return tags
