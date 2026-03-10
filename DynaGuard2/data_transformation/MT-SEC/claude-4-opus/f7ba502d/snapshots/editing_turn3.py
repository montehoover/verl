import re

def safe_eval_expression(expr: str) -> float:
    # Check if the expression contains only allowed characters
    # Allow digits, decimal points, operators (+, -, *, /, %), parentheses, and spaces
    if not re.match(r'^[0-9+\-*/().\s%]+$', expr):
        raise ValueError("Expression contains invalid characters")
    
    # Check for empty expression
    if not expr.strip():
        raise ValueError("Empty expression")
    
    # Additional safety checks
    # Check for invalid patterns like multiple operators in a row
    if re.search(r'[+\-*/]{2,}', expr):
        raise ValueError("Invalid operator sequence")
    
    # Check for invalid decimal points
    if re.search(r'\d*\.\d*\.', expr):
        raise ValueError("Invalid decimal number")
    
    try:
        # Use eval with restricted namespace for safety
        # Only allow basic math operations
        allowed_names = {
            '__builtins__': None,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
        }
        
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        
        # Ensure the result is a number
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a number")
        
        return float(result)
    
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")
