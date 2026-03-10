import re

def evaluate_user_expression(expression: str):
    """
    Safely calculates the result of a mathematical expression provided by the user.
    
    Args:
        expression: str, the string containing the user's mathematical expression
        
    Returns:
        The result of evaluating the mathematical expression
        
    Raises:
        ValueError: if the input contains invalid characters or if the expression is incorrect
    """
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    # Define allowed characters: digits, operators, parentheses, and decimal point
    allowed_pattern = re.compile(r'^[0-9+\-*/().\s]+$')
    
    # Check if expression contains only allowed characters
    if not allowed_pattern.match(expression):
        raise ValueError("Expression contains invalid characters")
    
    # Check for empty expression
    if not expression:
        raise ValueError("Expression is empty")
    
    # Additional safety checks
    # Check for dangerous patterns like double operators
    dangerous_patterns = [
        r'[+\-*/]{2,}',  # Multiple operators in a row (except for negative numbers)
        r'^[*/]',        # Expression starting with * or /
        r'[*/]$',        # Expression ending with * or /
        r'\(\)',         # Empty parentheses
        r'[+\-*/]\)',    # Operator before closing parenthesis
        r'\([+*/]',      # Opening parenthesis followed by operator (except minus)
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expression):
            raise ValueError("Invalid expression format")
    
    try:
        # Use eval with restricted namespace for safety
        # Only allow basic arithmetic operations
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except (SyntaxError, ZeroDivisionError, ArithmeticError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
