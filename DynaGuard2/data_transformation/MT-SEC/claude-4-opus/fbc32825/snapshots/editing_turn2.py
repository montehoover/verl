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

def capture_text_parts(input_text):
    """
    Captures text both outside and inside parentheses from a given string.
    
    Args:
        input_text (str): The input string to process
        
    Returns:
        tuple or None: A tuple containing (text_outside, text_inside) if pattern matches, None otherwise
    """
    match = re.search(r'^([^(]*)\(([^)]+)\)(.*)$', input_text)
    if match:
        outside_text = match.group(1) + match.group(3)
        inside_text = match.group(2)
        return (outside_text.strip(), inside_text)
    return None
