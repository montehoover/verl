import re

def extract_inside_parentheses(input_text):
    """
    Extracts text within parentheses from a given string.
    
    Args:
        input_text (str): The input string to search for text in parentheses
        
    Returns:
        str or None: The text inside parentheses if found, None otherwise
    """
    match = re.search(r'\(([^)]+)\)', input_text)
    if match:
        return match.group(1)
    return None
