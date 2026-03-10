def calculate_expression(expr_input):
    """
    Evaluates user-provided mathematical expressions in Python syntax.
    Returns the result of the evaluation or 'Computation Error!' if something goes wrong.
    """
    try:
        # Check for potentially dangerous operations
        dangerous_keywords = ['import', '__', 'exec', 'eval', 'open', 'file', 'input', 'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr', 'type', 'id', 'help', 'repr', 'ascii', 'ord', 'chr', 'bin', 'oct', 'hex', 'format', 'bytes', 'bytearray', 'memoryview', 'frozenset', 'property', 'classmethod', 'staticmethod', 'super', 'object', 'print', 'quit', 'exit', 'copyright', 'credits', 'license']
        
        expr_lower = expr_input.lower()
        for keyword in dangerous_keywords:
            if keyword in expr_lower:
                return "Computation Error!"
        
        # Check for function calls (parentheses after alphanumeric characters)
        import re
        if re.search(r'[a-zA-Z_]\w*\s*\(', expr_input):
            return "Computation Error!"
        
        # Only allow basic mathematical operations and numbers
        allowed_chars = set('0123456789+-*/().% ')
        if not all(c in allowed_chars for c in expr_input):
            return "Computation Error!"
        
        # Evaluate the expression
        result = eval(expr_input)
        
        # Return the result
        return result
        
    except:
        return "Computation Error!"
