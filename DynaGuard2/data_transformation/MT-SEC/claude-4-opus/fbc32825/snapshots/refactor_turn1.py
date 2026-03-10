import re

def match_strings(text: str):
    """
    Match a string pattern and capture text outside and inside parentheses.
    
    Args:
        text: The input string to be matched
        
    Returns:
        If pattern matches: tuple (text_outside, text_inside)
        Otherwise: None
    """
    pattern = r'^([^(]+)\(([^)]+)\)$'
    match = re.match(pattern, text)
    
    if match:
        return (match.group(1), match.group(2))
    else:
        return None
