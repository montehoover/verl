import re

def safe_eval_expression(expr: str):
    """
    Safely calculates the result of a mathematical expression provided by the user.
    
    Args:
        expr: str, the string containing the user's mathematical expression
        
    Returns:
        The result of evaluating the mathematical expression
        
    Raises:
        ValueError: if the input contains invalid characters or if the expression is incorrect
    """
    # Remove whitespace
    expr = expr.replace(" ", "")
    
    # Check for empty expression
    if not expr:
        raise ValueError("Empty expression")
    
    # Define allowed characters: digits, operators, parentheses, and decimal point
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expr):
        raise ValueError("Expression contains invalid characters")
    
    # Additional security checks
    # Check for dangerous patterns
    dangerous_patterns = [
        r'__',  # double underscore (could access special attributes)
        r'import',  # import statements
        r'eval',  # eval function
        r'exec',  # exec function
        r'[a-zA-Z]',  # any letters (except e for scientific notation, but we're not supporting that)
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expr):
            raise ValueError("Expression contains invalid characters")
    
    # Check for balanced parentheses
    paren_count = 0
    for char in expr:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        if paren_count < 0:
            raise ValueError("Unbalanced parentheses")
    
    if paren_count != 0:
        raise ValueError("Unbalanced parentheses")
    
    # Check for invalid operator sequences
    invalid_sequences = [
        r'[+\-*/]{2,}',  # Multiple operators in a row (except for cases like '+-')
        r'^[*/]',  # Expression starting with * or /
        r'[+\-*/]$',  # Expression ending with an operator
        r'\(\)',  # Empty parentheses
        r'[+\-*/]\)',  # Operator directly before closing parenthesis
        r'\([+*/]',  # Opening parenthesis followed by * or /
    ]
    
    for pattern in invalid_sequences:
        if re.search(pattern, expr):
            raise ValueError("Invalid expression syntax")
    
    # Special case: allow negative numbers after operators or at start
    expr_check = re.sub(r'(^|[+\-*/\(])-', r'\1', expr)
    if re.search(r'[+\-*/]{2,}', expr_check):
        raise ValueError("Invalid expression syntax")
    
    try:
        # Use eval with restricted namespace
        # Only allow built-in math operations, no functions or imports
        result = eval(expr, {"__builtins__": {}}, {})
        
        # Check for division by zero (eval might return inf)
        if isinstance(result, float) and (result == float('inf') or result == float('-inf')):
            raise ValueError("Division by zero")
            
        return result
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except Exception:
        raise ValueError("Invalid expression")
