import re

def is_safe_query(query):
    """
    Check if a string contains only numbers, basic arithmetic operators (+, -, *, /), and spaces.
    
    Args:
        query (str): The input string to validate
        
    Returns:
        bool: True if the input is safe, False otherwise
    """
    # Pattern matches only digits, arithmetic operators, spaces, and decimal points
    pattern = r'^[0-9+\-*/\s.]+$'
    
    if not query or not query.strip():
        return False
    
    return bool(re.match(pattern, query))
