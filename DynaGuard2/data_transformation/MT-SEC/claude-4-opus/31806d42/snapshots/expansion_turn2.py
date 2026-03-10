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


def apply_operator(num1, num2, operator):
    """
    Applies an arithmetic operator to two numbers.
    
    Args:
        num1 (float): First operand
        num2 (float): Second operand
        operator (str): The operator to apply (+, -, *, /)
        
    Returns:
        float: Result of the operation
        
    Raises:
        ValueError: If the operator is not supported
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


def compute_with_precedence(tokens):
    """
    Computes the result of an arithmetic expression with operator precedence.
    
    Args:
        tokens (list): List of numbers and operators in order
                      e.g., [2, '+', 3, '*', 4] represents "2 + 3 * 4"
        
    Returns:
        float: The result of the computation
        
    Raises:
        ValueError: If an unsupported operator is encountered
    """
    if not tokens:
        raise ValueError("Empty expression")
    
    # Handle single number case
    if len(tokens) == 1:
        return float(tokens[0])
    
    # First pass: handle multiplication and division
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 2 and tokens[i + 1] in ['*', '/']:
            left = float(tokens[i])
            operator = tokens[i + 1]
            right = float(tokens[i + 2])
            
            result = apply_operator(left, right, operator)
            
            # Replace the three tokens with the result
            tokens = tokens[:i] + [result] + tokens[i + 3:]
        else:
            i += 2
    
    # Second pass: handle addition and subtraction
    result = float(tokens[0])
    i = 1
    while i < len(tokens):
        operator = tokens[i]
        operand = float(tokens[i + 1])
        
        result = apply_operator(result, operand, operator)
        i += 2
    
    return result
