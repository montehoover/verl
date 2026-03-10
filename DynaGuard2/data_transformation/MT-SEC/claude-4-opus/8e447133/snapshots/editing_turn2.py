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
