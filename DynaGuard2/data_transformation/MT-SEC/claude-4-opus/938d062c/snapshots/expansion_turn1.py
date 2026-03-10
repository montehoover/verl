import re

def is_valid_expression(expression):
    """
    Validates if a string contains only numbers, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces.
    
    Args:
        expression (str): The expression string to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Pattern allows:
    # - digits (0-9)
    # - decimal points
    # - arithmetic operators: +, -, *, /
    # - parentheses: ( )
    # - spaces
    pattern = r'^[0-9\+\-\*/\(\)\.\s]+$'
    
    if not expression or not isinstance(expression, str):
        return False
    
    # Check if the expression matches the allowed pattern
    if not re.match(pattern, expression):
        return False
    
    # Additional validation to ensure proper structure
    # Check for empty parentheses
    if '()' in expression:
        return False
    
    # Check for multiple consecutive operators (except for negative numbers)
    if re.search(r'[\+\*/]{2,}', expression):
        return False
    
    # Check for operators at the beginning (except minus for negative numbers)
    if re.match(r'^[\+\*/]', expression):
        return False
    
    # Check for operators at the end
    if re.search(r'[\+\-\*/]$', expression):
        return False
    
    return True
