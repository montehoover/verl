import re

def match_strings(text):
    """
    Match a string pattern and extract text outside and inside parentheses.
    
    Args:
        text: str, the input string to be matched
        
    Returns:
        If pattern is matched, returns a tuple (text_outside, text_inside).
        Otherwise, returns None.
    """
    # Pattern to match text outside parentheses followed by text inside parentheses
    pattern = r'^([^(]+)\(([^)]+)\)$'
    
    match = re.match(pattern, text)
    
    if match:
        # Extract the two groups: text outside and text inside parentheses
        return (match.group(1), match.group(2))
    else:
        return None
