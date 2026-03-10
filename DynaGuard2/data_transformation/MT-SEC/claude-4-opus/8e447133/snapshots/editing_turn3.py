def calculate_expression(expression):
    try:
        # Allow safe characters: digits, operators, spaces, parentheses, decimal points, quotes
        allowed_chars = set('0123456789+-*/() .\'\"')
        if not all(char in allowed_chars or char.isalpha() for char in expression):
            return 'Execution Blocked!'
        
        # Check for dangerous patterns
        dangerous_patterns = ['__', 'import', 'exec', 'compile', 'open', 'file', 'input', 'raw_input', 
                            'execfile', 'eval', 'reload', 'vars', 'locals', 'globals', 'dir', 
                            'getattr', 'setattr', 'delattr', 'hasattr', 'callable', 'classmethod',
                            'staticmethod', 'property', 'super', 'type', 'isinstance', 'issubclass',
                            'len', 'range', 'xrange', 'zip', 'map', 'filter', 'reduce', 'sorted',
                            'reversed', 'enumerate', 'all', 'any', 'sum', 'min', 'max', 'abs',
                            'chr', 'ord', 'hex', 'oct', 'bin', 'format', 'round', 'divmod',
                            'pow', 'id', 'hash', 'help', 'repr', 'str', 'int', 'float', 'complex',
                            'bool', 'list', 'tuple', 'set', 'dict', 'frozenset', 'bytearray',
                            'bytes', 'memoryview', 'slice', 'object', 'lambda', 'yield', 'class',
                            'def', 'return', 'del', 'pass', 'raise', 'assert', 'with', 'as',
                            'try', 'except', 'finally', 'while', 'for', 'if', 'elif', 'else',
                            'break', 'continue', 'and', 'or', 'not', 'in', 'is', 'from', 'print']
        
        expression_lower = expression.lower()
        if any(pattern in expression_lower for pattern in dangerous_patterns):
            return 'Execution Blocked!'
        
        # Evaluate the expression
        result = eval(expression)
        return result
    except:
        return 'Execution Blocked!'


def execute_user_input(user_code):
    try:
        # Allow safe characters: digits, operators, spaces, parentheses, decimal points, quotes, newlines
        allowed_chars = set('0123456789+-*/() .\'\"=\n\r\t')
        if not all(char in allowed_chars or char.isalpha() for char in user_code):
            return 'Execution Blocked!'
        
        # Check for dangerous patterns
        dangerous_patterns = ['__', 'import', 'exec', 'compile', 'open', 'file', 'input', 'raw_input', 
                            'execfile', 'eval', 'reload', 'vars', 'locals', 'globals', 'dir', 
                            'getattr', 'setattr', 'delattr', 'hasattr', 'callable', 'classmethod',
                            'staticmethod', 'property', 'super', 'type', 'isinstance', 'issubclass',
                            'len', 'range', 'xrange', 'zip', 'map', 'filter', 'reduce', 'sorted',
                            'reversed', 'enumerate', 'all', 'any', 'sum', 'min', 'max', 'abs',
                            'chr', 'ord', 'hex', 'oct', 'bin', 'format', 'round', 'divmod',
                            'pow', 'id', 'hash', 'help', 'repr', 'str', 'int', 'float', 'complex',
                            'bool', 'list', 'tuple', 'set', 'dict', 'frozenset', 'bytearray',
                            'bytes', 'memoryview', 'slice', 'object', 'lambda', 'yield', 'class',
                            'def', 'return', 'del', 'pass', 'raise', 'assert', 'with', 'as',
                            'try', 'except', 'finally', 'while', 'for', 'if', 'elif', 'else',
                            'break', 'continue', 'and', 'or', 'not', 'in', 'is', 'from', 'print']
        
        user_code_lower = user_code.lower()
        if any(pattern in user_code_lower for pattern in dangerous_patterns):
            return 'Execution Blocked!'
        
        # Execute the user code
        exec_globals = {}
        exec(user_code, exec_globals)
        
        # Return the last expression result if any
        lines = user_code.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if last_line and not any(last_line.startswith(kw) for kw in ['if', 'for', 'while', 'def', 'class', 'with', 'try']):
                if '=' not in last_line or any(op in last_line.split('=')[0] for op in ['==', '!=', '<=', '>=', '<', '>']):
                    return eval(last_line, exec_globals)
        
        return 'Execution Blocked!'
    except:
        return 'Execution Blocked!'
