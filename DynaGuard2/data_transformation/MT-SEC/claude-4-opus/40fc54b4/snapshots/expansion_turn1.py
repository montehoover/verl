import re

def is_valid_expression(expression):
    """
    Check if a string is a valid arithmetic expression using only +, -, *, / operations.
    
    Args:
        expression: String representing the arithmetic expression
        
    Returns:
        Boolean indicating if the expression is valid
    """
    # Remove all whitespace
    expression = expression.replace(" ", "")
    
    # Empty string is not valid
    if not expression:
        return False
    
    # Check for invalid characters - only allow digits, operators, parentheses, and decimal points
    if not re.match(r'^[0-9+\-*/().\s]+$', expression):
        return False
    
    # Check for empty parentheses
    if "()" in expression:
        return False
    
    # Check for consecutive operators
    if re.search(r'[+\-*/]{2,}', expression):
        return False
    
    # Check if expression starts or ends with an operator (except - at start)
    if re.match(r'^[+*/]', expression) or re.search(r'[+\-*/]$', expression):
        return False
    
    # Check for operators before/after parentheses incorrectly
    if re.search(r'\)[0-9.]', expression) or re.search(r'[0-9.]\(', expression):
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
    
    # Check for valid number format (no multiple decimal points in a number)
    numbers = re.findall(r'[0-9.]+', expression)
    for num in numbers:
        if num.count('.') > 1:
            return False
        if num == '.':
            return False
    
    # Try to evaluate to catch any remaining syntax errors
    try:
        # Replace the expression with a safe version for validation
        # This is just to check syntax, not to actually evaluate
        test_expr = expression
        # Ensure no division by zero for validation
        test_expr = re.sub(r'/\s*0+(?![0-9.])', '/1', test_expr)
        compile(test_expr, '<string>', 'eval')
        return True
    except:
        return False
