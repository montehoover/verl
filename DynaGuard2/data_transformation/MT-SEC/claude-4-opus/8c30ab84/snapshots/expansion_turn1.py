import re

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
