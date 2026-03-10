import re

def sanitize_expression(expression):
    """
    Check if a given string only contains numbers, basic arithmetic operators (+, -, *, /), 
    parentheses, and spaces.
    
    Args:
        expression (str): The expression to sanitize
        
    Returns:
        bool: True if the expression is safe, False otherwise
    """
    # Define allowed characters: digits, operators, parentheses, decimal points, and spaces
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if the expression matches the allowed pattern
    if not re.match(allowed_pattern, expression):
        return False
    
    # Additional checks to prevent common issues
    # Check for empty string
    if not expression.strip():
        return False
    
    # Check for multiple consecutive operators (except for negative numbers)
    if re.search(r'[+*/]{2,}', expression):
        return False
    
    # Check for operators at the beginning (except minus for negative numbers)
    if re.match(r'^[+*/]', expression.strip()):
        return False
    
    # Check for operators at the end
    if re.search(r'[+\-*/]$', expression.strip()):
        return False
    
    return True
