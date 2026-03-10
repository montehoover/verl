import re

def contains_html_tags(text):
    """
    Check if a given string contains any HTML tags.
    
    Args:
        text (str): The string to check for HTML tags
        
    Returns:
        bool: True if HTML tags are present, False otherwise
    """
    try:
        # Regular expression pattern to match HTML tags
        # Matches opening tags like <tag>, closing tags like </tag>, and self-closing tags like <tag/>
        html_pattern = r'<[^>]+>'
        
        # Search for HTML tags in the text
        match = re.search(html_pattern, text)
        
        # Return True if match found, False otherwise
        return match is not None
    except:
        # Return False if any exception occurs
        return False

def extract_first_html_tag(text):
    """
    Extract the first HTML tag found in a string.
    
    Args:
        text (str): The string to search for HTML tags
        
    Returns:
        str or None: The first HTML tag as a string if found, None otherwise
    """
    try:
        # Regular expression pattern to match HTML tags
        # Matches opening tags like <tag>, closing tags like </tag>, and self-closing tags like <tag/>
        html_pattern = r'<[^>]+>'
        
        # Search for the first HTML tag in the text
        match = re.search(html_pattern, text)
        
        # Return the matched tag if found, None otherwise
        if match:
            return match.group(0)
        else:
            return None
    except:
        # Return None if any exception occurs
        return None

def identify_html_tags(content):
    """
    Scan a provided string for all HTML tags using a regular expression.
    
    Args:
        content (str): The string to scan for HTML tags
        
    Returns:
        list: A list of all HTML tags identified in the given input
    """
    try:
        # Regular expression pattern to match HTML tags
        # Matches opening tags like <tag>, closing tags like </tag>, and self-closing tags like <tag/>
        html_pattern = r'<[^>]+>'
        
        # Find all HTML tags in the content
        tags = re.findall(html_pattern, content)
        
        # Return the list of tags
        return tags
    except:
        # Return empty list if any exception occurs
        return []
