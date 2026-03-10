import re

def validate_expression(expression):
    """
    Validates that a mathematical expression is safe for evaluation.
    Returns True if valid, False otherwise.
    
    Allowed: numbers, +, -, *, /, (, ), spaces, and decimal points
    """
    # Check if string is empty or None
    if not expression or not expression.strip():
        return False
    
    # Define allowed characters: digits, operators, parentheses, decimal points, and spaces
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expression):
        return False
    
    # Check for balanced parentheses
    paren_count = 0
    for char in expression:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        if paren_count < 0:
            return False
    
    if paren_count != 0:
        return False
    
    # Check for invalid patterns
    # No multiple operators in a row (except for negative numbers)
    if re.search(r'[+*/]{2,}', expression):
        return False
    
    # No operators at the beginning except minus
    if re.match(r'^[+*/]', expression):
        return False
    
    # No operators at the end
    if re.search(r'[+\-*/]$', expression.strip()):
        return False
    
    # No empty parentheses
    if '()' in expression:
        return False
    
    # No multiple decimal points in a single number
    if re.search(r'\d*\.\d*\.\d*', expression):
        return False
    
    # No division by zero (basic check)
    if re.search(r'/\s*0(?:\s|$|[+\-*/)])', expression):
        return False
    
    return True
