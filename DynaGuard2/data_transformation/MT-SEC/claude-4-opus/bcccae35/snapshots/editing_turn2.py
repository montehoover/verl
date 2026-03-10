import re

def count_html_tags(html_content):
    """
    Count the number of HTML tags in a given string.
    
    Args:
        html_content (str): The HTML content to analyze
        
    Returns:
        int: The count of HTML tags found
    """
    try:
        # Regular expression to match HTML tags (opening and closing)
        # This pattern matches tags like <tag>, </tag>, <tag attr="value">, etc.
        tag_pattern = r'<[^>]+>'
        
        # Find all matches
        tags = re.findall(tag_pattern, html_content)
        
        # Return the count
        return len(tags)
    except Exception:
        # If any exception occurs, return 0
        return 0

def extract_unique_tags(html_content):
    """
    Extract unique HTML tags from the given string.
    
    Args:
        html_content (str): The HTML content to analyze
        
    Returns:
        list: A list of unique tag names found
    """
    try:
        # Regular expression to match HTML tags and extract tag names
        # This pattern captures the tag name after < or </
        tag_pattern = r'</?(\w+)[^>]*>'
        
        # Find all tag names
        tag_names = re.findall(tag_pattern, html_content)
        
        # Return unique tags as a list
        return list(set(tag_names))
    except Exception:
        # If any exception occurs, return empty list
        return []
