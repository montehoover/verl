def evaluate_user_math(math_expr: str) -> str:
    try:
        # Define allowed names for safe evaluation
        allowed_names = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'len': len,
        }
        
        # Check for potentially dangerous operations
        dangerous_patterns = [
            '__', 'import', 'exec', 'eval', 'open',
            'file', 'input', 'compile', 'globals',
            'locals', 'vars', 'dir', 'getattr',
            'setattr', 'delattr', 'hasattr',
            'type', 'isinstance', 'issubclass',
            'callable', 'classmethod', 'staticmethod',
            'property', 'super', 'object', 'bytes',
            'bytearray', 'memoryview', 'print',
            'breakpoint', 'help', 'quit', 'exit',
            'license', 'credits', 'copyright'
        ]
        
        # Check if expression contains any dangerous patterns
        expr_lower = math_expr.lower()
        for pattern in dangerous_patterns:
            if pattern in expr_lower:
                return 'Processing Failed!'
        
        # Check for suspicious characters that might indicate code injection
        suspicious_chars = [';', '\\', '`', '\n', '\r', '\t']
        for char in suspicious_chars:
            if char in math_expr:
                return 'Processing Failed!'
        
        # Use compile to check syntax and restrict to expressions only
        try:
            compiled = compile(math_expr, '<string>', 'eval')
        except SyntaxError:
            return 'Processing Failed!'
        
        # Evaluate the expression with restricted globals
        result = eval(compiled, {"__builtins__": {}}, allowed_names)
        
        return str(result)
        
    except:
        return 'Processing Failed!'
