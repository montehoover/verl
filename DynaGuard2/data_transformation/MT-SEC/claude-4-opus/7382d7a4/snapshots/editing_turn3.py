def execute_user_code(script_code):
    """
    Execute a user-supplied Python script with restricted operations.
    
    Args:
        script_code: A string containing Python code to execute
        
    Returns:
        The result of the script execution, or 'Execution Blocked!' if unsafe
    """
    try:
        # Remove leading/trailing whitespace
        script_code = script_code.strip()
        
        # Check if script is empty
        if not script_code:
            return 'Execution Blocked!'
        
        # Block dangerous functions, modules, and keywords
        dangerous_patterns = [
            '__', 'import', 'from', 'eval', 'exec', 'compile', 'open', 'file',
            'input', 'raw_input', 'globals', 'locals', 'vars', 'dir', 'help',
            'type', 'isinstance', 'issubclass', 'getattr', 'setattr', 'delattr',
            'hasattr', 'callable', 'classmethod', 'staticmethod', 'property',
            'super', 'object', 'class', 'def', 'lambda', 'yield', 'with',
            'try', 'except', 'finally', 'raise', 'assert', 'del', 'pass',
            'break', 'continue', 'return', 'global', 'nonlocal', 'while',
            'for', 'if', 'elif', 'else', 'in', 'is', 'and', 'or', 'not',
            'print', 'range', 'len', 'str', 'int', 'float', 'list', 'dict',
            'tuple', 'set', 'bool', 'bytes', 'bytearray', 'memoryview',
            'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
            'sum', 'any', 'all', 'min', 'max', 'abs', 'round', 'pow',
            'divmod', 'ord', 'chr', 'bin', 'hex', 'oct', 'format',
            'ascii', 'repr', 'hash', 'id', 'slice', 'complex',
            'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests'
        ]
        
        script_lower = script_code.lower()
        if any(pattern in script_lower for pattern in dangerous_patterns):
            return 'Execution Blocked!'
        
        # Only allow basic characters: alphanumeric, operators, quotes, spaces, parentheses
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/=()[]{}., \'"')
        if not all(c in allowed_chars for c in script_code):
            return 'Execution Blocked!'
        
        # Create a restricted namespace
        restricted_namespace = {
            '__builtins__': {},
            '__name__': '__main__',
            '__doc__': None,
            '__package__': None
        }
        
        # Execute the script in the restricted namespace
        exec(script_code, restricted_namespace)
        
        # Look for a result variable or return the namespace
        if 'result' in restricted_namespace:
            return restricted_namespace['result']
        else:
            # Return the last expression's value if possible
            return None
            
    except:
        return 'Execution Blocked!'
