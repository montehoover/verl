import re

def find_html_tags(html_content):
    """
    Identify all HTML tags in a given string using regular expressions.
    
    Args:
        html_content (str): The HTML content to analyze
        
    Returns:
        list: A list containing all HTML tags found, including duplicates
    """
    # Regular expression to match HTML tags
    tag_pattern = r'<[^>]+>'
    
    # Find all matches and return them
    return re.findall(tag_pattern, html_content)
