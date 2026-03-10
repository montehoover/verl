import re

def parse_math_expression(expression):
    """
    Parse a mathematical expression and determine if it's valid and safe.
    
    Args:
        expression (str): The mathematical expression to parse
        
    Returns:
        bool: True if the expression is valid and safe, False otherwise
    """
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    # Check if empty
    if not expression:
        return False
    
    # Define allowed characters: digits, operators, parentheses, and decimal point
    allowed_pattern = r'^[0-9+\-*/()\.]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expression):
        return False
    
    # Check for invalid patterns
    # Multiple operators in a row (except for negative numbers)
    if re.search(r'[+*/]{2,}', expression):
        return False
    
    # Operators at the start (except minus)
    if re.match(r'^[+*/]', expression):
        return False
    
    # Operators at the end
    if re.search(r'[+\-*/]$', expression):
        return False
    
    # Empty parentheses
    if '()' in expression:
        return False
    
    # Check parentheses balance
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
    
    # Check for invalid decimal patterns
    # Split by operators and parentheses to check each number
    numbers = re.split(r'[+\-*/()]', expression)
    for num in numbers:
        if num:  # Skip empty strings
            # Check for multiple decimal points
            if num.count('.') > 1:
                return False
            # Check for decimal point at start or end
            if num.startswith('.') or num.endswith('.'):
                return False
    
    # Basic validation passed
    return True
