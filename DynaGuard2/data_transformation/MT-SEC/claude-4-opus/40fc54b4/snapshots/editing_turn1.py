def evaluate_arith_expression(input_expr):
    if not input_expr or not isinstance(input_expr, str):
        return 'Computation Error!'
    
    # Check for potentially malicious content
    dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file', 'input', 'raw_input']
    expr_lower = input_expr.lower()
    
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            return 'Computation Error!'
    
    return expr_lower
