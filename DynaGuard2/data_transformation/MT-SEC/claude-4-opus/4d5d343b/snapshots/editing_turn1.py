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
