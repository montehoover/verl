def evaluate_expression(expr_string):
    try:
        # Check for suspicious patterns
        forbidden_patterns = ['import', '__', 'exec', 'eval', 'open', 'file', 'input', 'raw_input', 
                            'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 
                            'delattr', 'hasattr', 'callable', 'classmethod', 'staticmethod', 
                            'property', 'type', 'isinstance', 'issubclass', 'super', 'object',
                            'lambda', 'def', 'class', 'for', 'while', 'if', 'else', 'elif',
                            'try', 'except', 'finally', 'raise', 'assert', 'with', 'as',
                            'yield', 'return', 'break', 'continue', 'pass', 'del']
        
        expr_lower = expr_string.lower()
        for pattern in forbidden_patterns:
            if pattern in expr_lower:
                return "Computation Error!"
        
        # Only allow basic math operations and numbers
        allowed_chars = set("0123456789+-*/.() \t\n")
        if not all(c in allowed_chars for c in expr_string):
            return "Computation Error!"
        
        # Evaluate the expression
        result = eval(expr_string, {"__builtins__": {}}, {})
        return result
        
    except:
        return "Computation Error!"
