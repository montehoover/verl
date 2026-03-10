import re

def validate_expression(expression):
    """
    Validates that a mathematical expression is safe for evaluation.
    Returns True if valid, False otherwise.
    
    Allowed: numbers, +, -, *, /, (, ), spaces, and decimal points
    """
    # Check if string is empty or None
    if not expression or not expression.strip():
        return False
    
    # Define allowed characters: digits, operators, parentheses, decimal points, and spaces
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expression):
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
    
    # Check for invalid patterns
    # No multiple operators in a row (except for negative numbers)
    if re.search(r'[+*/]{2,}', expression):
        return False
    
    # No operators at the beginning except minus
    if re.match(r'^[+*/]', expression):
        return False
    
    # No operators at the end
    if re.search(r'[+\-*/]$', expression.strip()):
        return False
    
    # No empty parentheses
    if '()' in expression:
        return False
    
    # No multiple decimal points in a single number
    if re.search(r'\d*\.\d*\.\d*', expression):
        return False
    
    # No division by zero (basic check)
    if re.search(r'/\s*0(?:\s|$|[+\-*/)])', expression):
        return False
    
    return True


def safe_evaluate(expression):
    """
    Safely evaluates a mathematical expression and returns the result.
    Returns an error message if evaluation fails.
    """
    try:
        # Create a restricted namespace with only math functions we want to allow
        safe_dict = {
            '__builtins__': None,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
        }
        
        # Additional validation before eval
        if not validate_expression(expression):
            return "Error: Invalid expression"
        
        # Evaluate the expression
        result = eval(expression, safe_dict)
        
        # Check for infinity or NaN
        if isinstance(result, float):
            if result == float('inf') or result == float('-inf'):
                return "Error: Result is infinity"
            if result != result:  # NaN check
                return "Error: Result is not a number"
        
        return result
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError:
        return "Error: Invalid mathematical operation"
    except SyntaxError:
        return "Error: Invalid expression syntax"
    except Exception:
        return "Error: Computation problem"


def evaluate_expression(expr_string):
    """
    Processes a string input to compute and return the result.
    Returns 'Computation Error!' if expression is unsafe or evaluation fails.
    """
    # First validate the expression
    if not validate_expression(expr_string):
        return 'Computation Error!'
    
    # Then evaluate it safely
    result = safe_evaluate(expr_string)
    
    # Check if the result is an error message
    if isinstance(result, str) and result.startswith("Error:"):
        return 'Computation Error!'
    
    return result
