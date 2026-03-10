import re
import ast

def validate_expression(expression):
    """
    Validates if a string contains only digits, spaces, and basic math operators (+, -, *, /).
    
    Args:
        expression (str): The expression string to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Define the pattern for valid characters: digits, spaces, and basic operators
    valid_pattern = r'^[0-9\s+\-*/]+$'
    
    # Check if the expression matches the pattern
    if not re.match(valid_pattern, expression):
        return False
    
    # Additional check to ensure the expression is not empty or only whitespace
    if not expression or expression.isspace():
        return False
    
    # Check that the expression contains at least one digit
    if not any(char.isdigit() for char in expression):
        return False
    
    return True


def resolve_expression(tokens):
    """
    Computes the result of a mathematical expression given as a list of numbers and operators.
    Follows standard operator precedence (*, / before +, -).
    
    Args:
        tokens (list): List of numbers (int/float) and operators (str)
        
    Returns:
        float: The computed result
        
    Raises:
        ValueError: If an unsupported operator is encountered
    """
    # Create a copy to avoid modifying the original list
    tokens = tokens.copy()
    
    # Validate operators
    valid_operators = {'+', '-', '*', '/'}
    for i in range(1, len(tokens), 2):
        if tokens[i] not in valid_operators:
            raise ValueError(f"Unsupported operator: {tokens[i]}")
    
    # First pass: handle multiplication and division (left to right)
    i = 1
    while i < len(tokens):
        if tokens[i] == '*':
            result = tokens[i-1] * tokens[i+1]
            tokens = tokens[:i-1] + [result] + tokens[i+2:]
        elif tokens[i] == '/':
            if tokens[i+1] == 0:
                raise ValueError("Division by zero")
            result = tokens[i-1] / tokens[i+1]
            tokens = tokens[:i-1] + [result] + tokens[i+2:]
        else:
            i += 2
    
    # Second pass: handle addition and subtraction (left to right)
    i = 1
    while i < len(tokens):
        if tokens[i] == '+':
            result = tokens[i-1] + tokens[i+1]
            tokens = tokens[:i-1] + [result] + tokens[i+2:]
        elif tokens[i] == '-':
            result = tokens[i-1] - tokens[i+1]
            tokens = tokens[:i-1] + [result] + tokens[i+2:]
        else:
            i += 2
    
    return tokens[0]


def compute_expression(expression_str):
    """
    Computes the result of a mathematical expression string.
    
    Args:
        expression_str (str): The expression string to compute
        
    Returns:
        float: The computed result
        
    Raises:
        ValueError: If the expression is invalid or cannot be computed
    """
    # Validate the expression
    if not validate_expression(expression_str):
        raise ValueError("Invalid expression")
    
    # Tokenize the expression
    tokens = []
    current_number = ""
    
    for char in expression_str:
        if char.isdigit() or char == '.':
            current_number += char
        elif char in '+-*/':
            if current_number:
                tokens.append(float(current_number))
                current_number = ""
            tokens.append(char)
        elif char.isspace():
            if current_number:
                tokens.append(float(current_number))
                current_number = ""
    
    # Add the last number if any
    if current_number:
        tokens.append(float(current_number))
    
    # Ensure we have a valid token structure (number operator number...)
    if len(tokens) == 0:
        raise ValueError("Empty expression")
    if len(tokens) % 2 == 0:
        raise ValueError("Invalid expression structure")
    
    # Compute and return the result
    return resolve_expression(tokens)
