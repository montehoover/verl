def calculate_expression(expression):
    """
    Calculate the result of a simple arithmetic expression or string operation.
    
    Args:
        expression: A string containing a simple arithmetic expression or string operation
        
    Returns:
        The result of the calculation/operation, or 'Execution Blocked!' if invalid/unsafe
    """
    try:
        # Remove leading/trailing whitespace
        expression = expression.strip()
        
        # Check if expression is empty
        if not expression:
            return 'Execution Blocked!'
        
        # Only allow digits, operators, parentheses, decimal points, spaces, quotes, and letters
        allowed_chars = set('0123456789+-*/()., \'\"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
        if not all(c in allowed_chars for c in expression):
            return 'Execution Blocked!'
        
        # Block dangerous patterns
        dangerous_patterns = ['import', '__', 'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input', 
                            'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr',
                            'type', 'isinstance', 'issubclass', 'callable', 'classmethod', 'staticmethod',
                            'property', 'super', 'object', 'class', 'lambda', 'def', 'return', 'yield',
                            'with', 'as', 'try', 'except', 'finally', 'raise', 'assert', 'del',
                            'from', 'global', 'nonlocal', 'pass', 'break', 'continue', 'if', 'elif',
                            'else', 'while', 'for', 'in', 'is', 'not', 'and', 'or', 'True', 'False',
                            'None', 'print', 'input', 'chr', 'ord', 'hex', 'bin', 'oct', 'bytes',
                            'bytearray', 'memoryview', 'frozenset', 'set', 'dict', 'list', 'tuple',
                            'range', 'slice', 'filter', 'map', 'zip', 'reversed', 'sorted', 'enumerate',
                            'all', 'any', 'sum', 'min', 'max', 'abs', 'round', 'pow', 'divmod',
                            'len', 'format', 'repr', 'str', 'int', 'float', 'complex', 'bool']
        
        # Check for dangerous patterns (case-insensitive)
        expression_lower = expression.lower()
        # Only check for actual dangerous patterns, not common string content
        actual_dangerous = ['import', '__', 'eval', 'exec', 'compile', 'open', 'file', 'globals', 'locals']
        for pattern in actual_dangerous:
            # Check if pattern appears as a separate word (not part of a string)
            if pattern in expression_lower and not (f"'{pattern}'" in expression or f'"{pattern}"' in expression):
                return 'Execution Blocked!'
        
        # Evaluate the expression
        result = eval(expression)
        return result
        
    except:
        return 'Execution Blocked!'
