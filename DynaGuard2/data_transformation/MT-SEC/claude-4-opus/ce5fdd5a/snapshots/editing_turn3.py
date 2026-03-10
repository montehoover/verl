def evaluate_expression(expr: str) -> str:
    if not expr or not isinstance(expr, str):
        return "Error!"
    
    # Whitelist of allowed characters and operations
    allowed_chars = set('0123456789+-*/() <>!=andornot.')
    
    # Check if expression contains only allowed characters
    if not all(c in allowed_chars or c.isspace() for c in expr):
        return "Error!"
    
    # Blacklist of dangerous keywords
    dangerous_patterns = ['import', '__', 'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input', 'print', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr']
    
    expr_lower = expr.lower()
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            return "Error!"
    
    try:
        # Evaluate the expression
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except:
        return "Error!"
