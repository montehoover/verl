def evaluate_expression(expression):
    """
    Evaluate simple arithmetic expressions given as strings.
    
    Args:
        expression: A string containing an arithmetic expression
        
    Returns:
        The result of the expression or 'Error!' if invalid/unsafe
    """
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check if expression contains only allowed characters
        allowed_chars = "0123456789+-*/()."
        allowed_words = ["True", "False", "and", "or", "not"]
        
        # Create a temporary expression to check characters
        temp_expression = expression
        for word in allowed_words:
            temp_expression = temp_expression.replace(word, "")
        
        if not all(char in allowed_chars for char in temp_expression):
            return 'Error!'
        
        # Check for empty expression
        if not expression:
            return 'Error!'
        
        # Evaluate the expression
        result = eval(expression)
        
        # Handle division by zero
        if isinstance(result, (int, float)) and (result == float('inf') or result == float('-inf')):
            return 'Error!'
        
        return result
        
    except:
        return 'Error!'


def evaluate_quiz_expression(quiz_expr):
    """
    Evaluate a user-provided Python expression for quiz context.
    
    Args:
        quiz_expr: A string containing a Python expression
        
    Returns:
        The result of the expression or an error message if invalid/dangerous
    """
    try:
        # Check for dangerous built-in functions and imports
        dangerous_patterns = ['import', '__import__', 'exec', 'eval', 'compile', 
                            'open', 'file', 'input', 'raw_input', 'execfile',
                            'getattr', 'setattr', 'delattr', 'globals', 'locals',
                            'vars', 'dir', '__', 'os', 'sys', 'subprocess']
        
        expr_lower = quiz_expr.lower()
        for pattern in dangerous_patterns:
            if pattern in expr_lower:
                return "Error: Unsafe operation detected"
        
        # Evaluate the expression with restricted globals
        allowed_names = {
            'abs': abs,
            'all': all,
            'any': any,
            'bin': bin,
            'bool': bool,
            'chr': chr,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'hex': hex,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'range': range,
            'reversed': reversed,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
            'True': True,
            'False': False,
            'None': None
        }
        
        result = eval(quiz_expr, {"__builtins__": {}}, allowed_names)
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"
