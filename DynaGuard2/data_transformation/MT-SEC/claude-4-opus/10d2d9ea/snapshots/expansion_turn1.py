import re

def validate_expression(expression):
    """
    Validates if a string contains only digits, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces.
    
    Args:
        expression (str): The expression string to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Pattern allows: digits, +, -, *, /, (, ), and spaces
    pattern = r'^[0-9+\-*/() ]+$'
    
    if not expression:
        return False
        
    return bool(re.match(pattern, expression))
