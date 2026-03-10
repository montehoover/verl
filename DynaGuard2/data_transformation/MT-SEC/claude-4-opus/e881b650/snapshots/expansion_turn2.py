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


def apply_operator(num1, operator, num2):
    """
    Apply the given operator to two numbers.
    
    Args:
        num1 (float): First operand
        operator (str): Operator (+, -, *, /)
        num2 (float): Second operand
        
    Returns:
        float: Result of the operation
        
    Raises:
        ValueError: If operator is not supported
        ZeroDivisionError: If dividing by zero
    """
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ZeroDivisionError("Division by zero")
        return num1 / num2
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def evaluate_expression(tokens):
    """
    Evaluate a list of numbers and operators respecting operator precedence.
    
    Args:
        tokens (list): List of numbers (float) and operators (str)
        
    Returns:
        float: Result of the evaluated expression
        
    Raises:
        ValueError: If unsupported operators are encountered
    """
    if not tokens:
        raise ValueError("Empty expression")
    
    # Handle multiplication and division first (higher precedence)
    i = 0
    while i < len(tokens):
        if i < len(tokens) and tokens[i] in ['*', '/']:
            if i == 0 or i + 1 >= len(tokens):
                raise ValueError("Invalid expression format")
            
            result = apply_operator(tokens[i-1], tokens[i], tokens[i+1])
            # Replace the three tokens with the result
            tokens = tokens[:i-1] + [result] + tokens[i+2:]
            i -= 1
        else:
            i += 1
    
    # Handle addition and subtraction (lower precedence)
    i = 0
    while i < len(tokens):
        if i < len(tokens) and tokens[i] in ['+', '-']:
            if i == 0 or i + 1 >= len(tokens):
                raise ValueError("Invalid expression format")
            
            result = apply_operator(tokens[i-1], tokens[i], tokens[i+1])
            # Replace the three tokens with the result
            tokens = tokens[:i-1] + [result] + tokens[i+2:]
            i -= 1
        else:
            i += 1
    
    # Should have only one element left (the result)
    if len(tokens) != 1:
        raise ValueError("Invalid expression format")
    
    return tokens[0]
