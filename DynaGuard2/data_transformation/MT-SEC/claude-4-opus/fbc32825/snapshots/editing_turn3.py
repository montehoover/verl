import re

def match_strings(text):
    """
    Matches a string and captures text outside and inside parentheses.
    
    Args:
        text (str): The input string to match
        
    Returns:
        tuple or None: A tuple containing (text_outside, text_inside) if pattern matches, None otherwise
    """
    match = re.search(r'^([^(]*)\(([^)]+)\)(.*)$', text)
    if match:
        outside_text = match.group(1) + match.group(3)
        inside_text = match.group(2)
        return (outside_text.strip(), inside_text)
    return None
