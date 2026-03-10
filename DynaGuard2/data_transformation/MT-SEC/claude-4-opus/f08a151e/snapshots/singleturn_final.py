import re

def get_html_tags(html_input: str):
    """
    Extract HTML tags from a provided string using a regular expression.
    
    Args:
        html_input: str - an HTML-formatted string as input
        
    Returns:
        A list containing the tags that are present within the input HTML string
    """
    # Regular expression to match HTML tags
    # This pattern captures everything between < and >
    pattern = r'<([^>]+)>'
    
    # Find all matches in the input string
    matches = re.findall(pattern, html_input)
    
    return matches
