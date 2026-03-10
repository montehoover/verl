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
        # Matches patterns like <tag>, </tag>, <tag attr="value">, <tag/>
        tag_pattern = r'<[^>]+>'
        
        # Find all matches
        matches = re.findall(tag_pattern, html_content)
        
        # Return the count
        return len(matches)
    except:
        # If any exception occurs, return 0
        return 0

def extract_unique_html_tags(html_content):
    """
    Extract unique HTML tags from a given string.
    
    Args:
        html_content (str): The HTML content to analyze
        
    Returns:
        list: A list of unique tags found in the input
    """
    try:
        # Regular expression to match HTML tags
        tag_pattern = r'<(/?)(\w+)[^>]*>'
        
        # Find all matches
        matches = re.findall(tag_pattern, html_content)
        
        # Extract tag names (without < > and attributes)
        tags = []
        for match in matches:
            tag_name = match[1].lower()
            tags.append(tag_name)
        
        # Return unique tags
        return list(set(tags))
    except:
        # If any exception occurs, return empty list
        return []
