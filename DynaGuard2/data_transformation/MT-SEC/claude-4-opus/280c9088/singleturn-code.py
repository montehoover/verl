def calculate_expression(expr_input: str) -> str:
    """
    Safely evaluates a mathematical expression provided as a string.
    
    Args:
        expr_input: A string containing a basic mathematical expression
        
    Returns:
        The result of the evaluation as a string, or 'Computation Error!' on failure
    """
    try:
        # Define allowed names for eval (only basic math operations)
        allowed_names = {
            '__builtins__': {}  # No built-in functions allowed
        }
        
        # Check for suspicious characters or patterns
        suspicious_patterns = [
            '__', 'import', 'exec', 'eval', 'open', 'file', 'input', 
            'raw_input', 'compile', 'globals', 'locals', 'vars',
            ';', 'lambda', 'def', 'class', 'for', 'while', 'if',
            '[', ']', '{', '}', '=', ':', ',', 'print'
        ]
        
        for pattern in suspicious_patterns:
            if pattern in expr_input:
                return "Computation Error!"
        
        # Only allow digits, spaces, parentheses, and basic operators
        import re
        if not re.match(r'^[\d\s\+\-\*/\(\)\.]+$', expr_input):
            return "Computation Error!"
        
        # Evaluate the expression with restricted namespace
        result = eval(expr_input, {"__builtins__": {}}, {})
        
        # Convert result to string
        return str(result)
        
    except:
        return "Computation Error!"
