def _check_dangerous_patterns(math_expr: str) -> bool:
    """
    Check if expression contains dangerous patterns that could pose security risks.
    
    Args:
        math_expr: The mathematical expression to check
        
    Returns:
        True if dangerous patterns are found, False otherwise
    """
    # List of potentially dangerous keywords that could enable code injection
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
    
    # Convert to lowercase for case-insensitive matching
    expr_lower = math_expr.lower()
    
    # Check each pattern
    for pattern in dangerous_patterns:
        if pattern in expr_lower:
            return True
    
    return False


def _check_suspicious_characters(math_expr: str) -> bool:
    """
    Check if expression contains suspicious characters that might indicate injection attempts.
    
    Args:
        math_expr: The mathematical expression to check
        
    Returns:
        True if suspicious characters are found, False otherwise
    """
    # Characters that could be used for code injection or breaking out of expressions
    suspicious_chars = [';', '\\', '`', '\n', '\r', '\t']
    
    # Check each character
    for char in suspicious_chars:
        if char in math_expr:
            return True
    
    return False


def _validate_syntax(math_expr: str):
    """
    Validate expression syntax and compile it for safe evaluation.
    
    Args:
        math_expr: The mathematical expression to validate
        
    Returns:
        Compiled code object if valid, None if syntax error
    """
    try:
        # Compile as eval mode to ensure it's a single expression, not statements
        return compile(math_expr, '<string>', 'eval')
    except SyntaxError:
        return None


def _execute_calculation(compiled_expr, allowed_names: dict) -> str:
    """
    Execute the compiled expression in a restricted environment.
    
    Args:
        compiled_expr: The compiled expression to execute
        allowed_names: Dictionary of allowed function names for evaluation
        
    Returns:
        String representation of the result, or None if execution fails
    """
    try:
        # Execute with empty builtins to prevent access to dangerous functions
        result = eval(compiled_expr, {"__builtins__": {}}, allowed_names)
        return str(result)
    except:
        return None


def evaluate_user_math(math_expr: str) -> str:
    """
    Safely evaluate a mathematical expression provided by the user.
    
    This function implements multiple security checks to prevent code injection
    and only allows basic mathematical operations and safe functions.
    
    Args:
        math_expr: A string representing the user-supplied expression to evaluate
        
    Returns:
        The outcome of the evaluated expression as a string, or 'Processing Failed!'
        if there is any error or security issue detected
    """
    try:
        # Define whitelist of safe mathematical functions
        allowed_names = {
            'abs': abs,      # Absolute value
            'round': round,  # Rounding
            'min': min,      # Minimum value
            'max': max,      # Maximum value
            'sum': sum,      # Sum of values
            'pow': pow,      # Power function
            'len': len,      # Length function
        }
        
        # Early return if dangerous patterns detected
        if _check_dangerous_patterns(math_expr):
            return 'Processing Failed!'
        
        # Early return if suspicious characters found
        if _check_suspicious_characters(math_expr):
            return 'Processing Failed!'
        
        # Validate and compile the expression
        compiled = _validate_syntax(math_expr)
        
        # Early return if compilation failed
        if compiled is None:
            return 'Processing Failed!'
        
        # Execute the calculation in restricted environment
        result = _execute_calculation(compiled, allowed_names)
        
        # Early return if execution failed
        if result is None:
            return 'Processing Failed!'
        
        return result
        
    except:
        # Catch any unexpected errors
        return 'Processing Failed!'
