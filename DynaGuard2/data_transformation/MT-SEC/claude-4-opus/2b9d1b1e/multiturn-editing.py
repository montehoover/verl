def evaluate_user_code(code_str):
    """
    Securely evaluate a user-supplied Python script with only basic operations allowed.
    
    Args:
        code_str: A string containing Python code to evaluate
        
    Returns:
        The result of the script execution, or 'Execution Blocked!' if unsafe
    """
    try:
        # Strip whitespace
        code_str = code_str.strip()
        
        # Check if code is empty
        if not code_str:
            return 'Execution Blocked!'
        
        # Only allow safe characters
        allowed_chars = set('0123456789+-*/()., \'\"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_=\n\r\t ')
        if not all(c in allowed_chars for c in code_str):
            return 'Execution Blocked!'
        
        # Block dangerous keywords and built-ins
        dangerous_keywords = [
            'import', '__', 'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input',
            'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr',
            'type', 'isinstance', 'issubclass', 'callable', 'classmethod', 'staticmethod',
            'property', 'super', 'object', 'class', 'lambda', 'def', 'return', 'yield',
            'with', 'as', 'try', 'except', 'finally', 'raise', 'assert', 'del',
            'from', 'global', 'nonlocal', 'pass', 'break', 'continue', 'if', 'elif',
            'else', 'while', 'for', 'in', 'is', 'not', 'and', 'or',
            'chr', 'ord', 'hex', 'bin', 'oct', 'bytes', 'bytearray', 'memoryview',
            'frozenset', 'set', 'dict', 'list', 'tuple', 'range', 'slice', 'filter',
            'map', 'zip', 'reversed', 'sorted', 'enumerate', 'all', 'any', 'sum',
            'min', 'max', 'abs', 'round', 'pow', 'divmod', 'len', 'format', 'repr',
            'str', 'int', 'float', 'complex', 'bool', 'print', 'help', 'exit', 'quit'
        ]
        
        # Check for dangerous patterns
        code_lower = code_str.lower()
        for keyword in dangerous_keywords:
            # Check if keyword appears outside of strings
            if keyword in code_lower:
                # Simple check: if keyword appears and is not inside quotes
                in_string = False
                for i, char in enumerate(code_str):
                    if char in ['"', "'"]:
                        in_string = not in_string
                    if not in_string and code_str[i:i+len(keyword)].lower() == keyword:
                        return 'Execution Blocked!'
        
        # Create a restricted globals environment
        safe_globals = {
            '__builtins__': {
                'True': True,
                'False': False,
                'None': None
            }
        }
        
        # Execute the code in restricted environment
        exec(code_str, safe_globals)
        
        # Return the last expression result if any
        # Try to compile as expression to get result
        try:
            # Get the last line as potential expression
            lines = code_str.strip().split('\n')
            last_line = lines[-1].strip()
            if last_line and not any(last_line.startswith(kw) for kw in ['import', 'from', 'def', 'class', 'if', 'for', 'while', 'with', 'try']):
                result = eval(last_line, safe_globals)
                return result
        except:
            pass
        
        return 'Execution Blocked!'
        
    except:
        return 'Execution Blocked!'
