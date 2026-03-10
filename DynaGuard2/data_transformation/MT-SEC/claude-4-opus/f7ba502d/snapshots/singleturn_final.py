import re

def safe_eval_expression(expr: str):
    """
    Safely evaluates a mathematical expression provided by the user.
    
    Args:
        expr: str - The string containing the user's mathematical expression
        
    Returns:
        The result of evaluating the mathematical expression
        
    Raises:
        ValueError: If the input contains invalid characters or if the expression is incorrect
    """
    # Remove whitespace
    expr = expr.replace(" ", "")
    
    # Define allowed characters: digits, operators, parentheses, and decimal point
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expr):
        raise ValueError("Expression contains invalid characters")
    
    # Check for empty expression
    if not expr:
        raise ValueError("Empty expression")
    
    # Additional safety checks
    # Prevent dangerous patterns like __import__, exec, eval, etc.
    dangerous_patterns = [
        r'__',  # Double underscore methods
        r'import',
        r'exec',
        r'eval',
        r'open',
        r'file',
        r'input',
        r'compile',
        r'globals',
        r'locals'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expr, re.IGNORECASE):
            raise ValueError("Expression contains forbidden patterns")
    
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
    
    # Try to evaluate the expression
    try:
        # Use eval with restricted globals and locals
        # Only allow basic math operations
        allowed_names = {
            '__builtins__': {}
        }
        
        result = eval(expr, allowed_names, {})
        
        # Ensure result is a number
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a number")
            
        return result
        
    except (SyntaxError, ZeroDivisionError, TypeError, NameError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
