def evaluate_arith_expression(input_expr: str) -> str:
    if not input_expr or not isinstance(input_expr, str):
        return 'Computation Error!'
    
    # Check for potentially dangerous elements
    dangerous_patterns = ['import', '__', 'exec', 'eval', 'open', 'file', 'input', 'raw_input', 
                         'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 
                         'delattr', 'hasattr', 'callable', 'classmethod', 'staticmethod', 
                         'property', 'type', 'isinstance', 'issubclass', 'super', 'object',
                         'lambda', 'def', 'class', 'for', 'while', 'if', 'try', 'except',
                         'raise', 'assert', 'del', 'pass', 'break', 'continue', 'return',
                         'yield', 'with', 'as', 'finally', 'elif', 'else', 'from', 'in',
                         'is', 'not', 'and', 'or', 'global', 'nonlocal', 'await', 'async']
    
    expr_lower = input_expr.lower()
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            return 'Computation Error!'
    
    # Only allow numbers, basic operators, parentheses, and whitespace
    allowed_chars = set('0123456789+-*/()., \t\n')
    if not all(c in allowed_chars for c in input_expr):
        return 'Computation Error!'
    
    try:
        # Evaluate the arithmetic expression
        result = eval(input_expr)
        return str(result)
    except:
        return 'Computation Error!'
