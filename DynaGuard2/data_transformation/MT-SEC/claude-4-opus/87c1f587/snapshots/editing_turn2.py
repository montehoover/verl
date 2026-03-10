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

def list_unique_html_tags(html_string):
    """
    List unique HTML tags found in a given string.
    
    Args:
        html_string (str): The HTML string to analyze
        
    Returns:
        list: A list of unique HTML tags found
    """
    try:
        # Regular expression to match HTML tags and extract tag names
        # Captures the tag name after < or </
        tag_pattern = r'</?([a-zA-Z][a-zA-Z0-9]*)[^>]*>'
        
        # Find all tag names
        tag_names = re.findall(tag_pattern, html_string)
        
        # Convert to lowercase and remove duplicates
        unique_tags = list(set(tag.lower() for tag in tag_names))
        
        # Sort the list for consistent output
        unique_tags.sort()
        
        return unique_tags
    except:
        # Return empty list if any exception occurs
        return []
