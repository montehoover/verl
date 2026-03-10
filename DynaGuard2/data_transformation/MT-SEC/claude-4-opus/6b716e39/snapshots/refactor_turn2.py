def _check_dangerous_patterns(math_expr: str) -> bool:
    """Check if expression contains dangerous patterns."""
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
    
    expr_lower = math_expr.lower()
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            return True
    return False


def _check_suspicious_characters(math_expr: str) -> bool:
    """Check if expression contains suspicious characters."""
    suspicious_chars = [';', '\\', '`', '\n', '\r', '\t']
    for char in suspicious_chars:
        if char in math_expr:
            return True
    return False


def _validate_syntax(math_expr: str):
    """Validate expression syntax and compile it."""
    try:
        return compile(math_expr, '<string>', 'eval')
    except SyntaxError:
        return None


def _execute_calculation(compiled_expr, allowed_names: dict) -> str:
    """Execute the compiled expression safely."""
    try:
        result = eval(compiled_expr, {"__builtins__": {}}, allowed_names)
        return str(result)
    except:
        return None


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
        
        # Security check pipeline
        if _check_dangerous_patterns(math_expr):
            return 'Processing Failed!'
        
        if _check_suspicious_characters(math_expr):
            return 'Processing Failed!'
        
        # Validation and compilation
        compiled = _validate_syntax(math_expr)
        if compiled is None:
            return 'Processing Failed!'
        
        # Calculation
        result = _execute_calculation(compiled, allowed_names)
        if result is None:
            return 'Processing Failed!'
        
        return result
        
    except:
        return 'Processing Failed!'
