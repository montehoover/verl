def calculate_expression(expr_input: str) -> str:
    """
    Evaluates a mathematical expression provided as a string.
    
    Args:
        expr_input: A string containing a basic mathematical expression
        
    Returns:
        The result of the evaluation as a string, or 'Computation Error!' if failed
    """
    try:
        # Check for suspicious characters or keywords
        forbidden_items = [
            'import', 'exec', 'eval', 'open', 'file', 'input', 'raw_input',
            '__', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
            'delattr', 'compile', 'chr', 'ord', 'hex', 'oct', 'bin',
            'breakpoint', 'help', 'quit', 'exit', 'license', 'credits',
            'copyright', ';', '\\', '[', ']', '{', '}', '=', ':', ',',
            'lambda', 'def', 'class', 'if', 'else', 'elif', 'for', 'while',
            'try', 'except', 'finally', 'with', 'as', 'assert', 'del',
            'global', 'nonlocal', 'yield', 'return', 'pass', 'break',
            'continue', 'raise', 'from', 'is', 'in', 'not', 'and', 'or'
        ]
        
        expr_lower = expr_input.lower()
        for item in forbidden_items:
            if item in expr_lower:
                return 'Computation Error!'
        
        # Only allow specific characters: digits, operators, parentheses, decimal point, and whitespace
        allowed_chars = set('0123456789+-*/() .\n\t')
        if not all(c in allowed_chars for c in expr_input):
            return 'Computation Error!'
        
        # Check for empty expression
        if not expr_input.strip():
            return 'Computation Error!'
        
        # Evaluate the expression
        result = eval(expr_input)
        
        # Convert result to string
        return str(result)
        
    except:
        return 'Computation Error!'
