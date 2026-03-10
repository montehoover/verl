import re

def is_valid_expression(expression):
    """
    Check if a given string contains only numbers, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces.
    
    Args:
        expression (str): The expression string to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Pattern allows: digits, decimal points, operators (+, -, *, /), parentheses, and spaces
    pattern = r'^[0-9\+\-\*/\(\)\.\s]+$'
    
    if not expression:
        return False
    
    if not re.match(pattern, expression):
        return False
    
    # Additional validation to ensure proper decimal number format
    # Prevents cases like "..5" or "5.."
    if '..' in expression:
        return False
    
    # Check for balanced parentheses
    open_count = 0
    for char in expression:
        if char == '(':
            open_count += 1
        elif char == ')':
            open_count -= 1
            if open_count < 0:
                return False
    
    return open_count == 0
