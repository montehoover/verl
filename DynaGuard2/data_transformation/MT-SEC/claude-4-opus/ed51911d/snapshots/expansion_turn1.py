import re

def parse_placeholders(html_string):
    """
    Parse HTML string and extract all placeholder names formatted as {{...}}
    Handles nested placeholders by finding the outermost {{ }} pairs
    
    Args:
        html_string (str): HTML string containing placeholders
        
    Returns:
        list: List of placeholder names found in the HTML string
    """
    placeholders = []
    
    # Pattern to match {{...}} including nested braces
    # This finds the outermost {{ }} pairs
    pattern = r'\{\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}\}'
    
    matches = re.findall(pattern, html_string)
    
    for match in matches:
        # Check if this match contains nested placeholders
        nested_matches = re.findall(r'\{\{([^{}]+)\}\}', match)
        if nested_matches:
            placeholders.extend(nested_matches)
        else:
            placeholders.append(match.strip())
    
    return placeholders
