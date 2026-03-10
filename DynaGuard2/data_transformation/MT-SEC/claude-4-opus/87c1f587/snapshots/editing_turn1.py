import re

def count_html_tags(html_string):
    """
    Count the number of HTML tags in a given string.
    
    Args:
        html_string (str): The HTML string to analyze
        
    Returns:
        int: The count of HTML tags found
    """
    try:
        # Regular expression to match HTML tags (opening and closing)
        # Matches tags like <tag>, </tag>, <tag attr="value">, <tag/>
        tag_pattern = r'<[^>]+>'
        
        # Find all matches
        matches = re.findall(tag_pattern, html_string)
        
        # Return the count
        return len(matches)
    except:
        # Return 0 if any exception occurs
        return 0
