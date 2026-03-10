def evaluate_expression(expr):
    if not expr or not isinstance(expr, str):
        return "Error!"
    
    # Check for potentially dangerous characters/patterns
    dangerous_patterns = ['import', '__', 'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input']
    expr_lower = expr.lower()
    
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            return "Error!"
    
    return expr_lower
