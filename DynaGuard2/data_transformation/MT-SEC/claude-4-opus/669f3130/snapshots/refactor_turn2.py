def validate_expression(calc_string: str) -> bool:
    """Validate that the expression is safe to evaluate."""
    # Check for dangerous operations
    dangerous_keywords = ['import', '__', 'exec', 'eval', 'compile', 'open', 'file', 'input', 'raw_input', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr', 'type', 'isinstance', 'issubclass', 'callable', 'classmethod', 'staticmethod', 'property', 'super', 'object', 'print', 'help', 'quit', 'exit', 'copyright', 'credits', 'license']
    
    for keyword in dangerous_keywords:
        if keyword in calc_string.lower():
            return False
    
    # Check for function calls
    if '(' in calc_string and ')' in calc_string:
        return False
    
    # Only allow numbers, basic operators, spaces, and decimal points
    allowed_chars = '0123456789+-*/ .'
    for char in calc_string:
        if char not in allowed_chars:
            return False
    
    return True


def evaluate_expression(calc_string: str) -> str:
    """Evaluate a validated mathematical expression."""
    try:
        result = eval(calc_string)
        return str(result)
    except:
        raise ValueError("Failed to evaluate expression")


def exec_calculation(calc_string: str) -> str:
    try:
        if not validate_expression(calc_string):
            return 'Computation Error!'
        
        return evaluate_expression(calc_string)
        
    except:
        return 'Computation Error!'
