import re

def evaluate_user_expression(expression: str):
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    # Check for invalid characters
    if not re.match(r'^[0-9+\-*/().\s]+$', expression):
        raise ValueError("Invalid characters in expression")
    
    # Check for empty expression
    if not expression:
        raise ValueError("Empty expression")
    
    try:
        # Use eval with restricted namespace for safety
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except (SyntaxError, ZeroDivisionError, TypeError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
