import re

def is_valid_expression(expression):
    """
    Validates if a string contains only numbers, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces.
    
    Args:
        expression (str): The expression to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Pattern allows: digits, decimal points, operators (+, -, *, /), parentheses, and spaces
    pattern = r'^[0-9\.\+\-\*/\(\)\s]+$'
    
    if not expression or not isinstance(expression, str):
        return False
    
    # Check if the expression matches the allowed pattern
    if not re.match(pattern, expression):
        return False
    
    # Additional validation to prevent empty expressions or only spaces
    if not expression.strip():
        return False
    
    return True
