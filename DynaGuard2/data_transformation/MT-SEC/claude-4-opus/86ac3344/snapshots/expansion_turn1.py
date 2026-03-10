def validate_math_expression(expression):
    """
    Validates if a mathematical expression contains only safe characters.
    
    Args:
        expression (str): The mathematical expression to validate
        
    Returns:
        bool: True if the expression contains only numbers, operators (+, -, *, /), 
              and spaces. False otherwise.
    """
    allowed_chars = set('0123456789+-*/ ')
    return all(char in allowed_chars for char in expression)
