def evaluate_arith_expression(input_expr):
    if not input_expr or not isinstance(input_expr, str):
        return 'Computation Error!'
    
    # Check for potentially malicious content
    dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file', 'input', 'raw_input']
    expr_lower = input_expr.lower()
    
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            return 'Computation Error!'
    
    # Handle specific prefixes
    if input_expr.startswith("TO_UPPER:"):
        content = input_expr[9:]  # Remove "TO_UPPER:" prefix
        return content.upper()
    elif input_expr.startswith("SLICE:"):
        content = input_expr[6:]  # Remove "SLICE:" prefix
        if len(content) > 5:
            return content[5:]
        else:
            return ''
    else:
        return 'Computation Error!'
