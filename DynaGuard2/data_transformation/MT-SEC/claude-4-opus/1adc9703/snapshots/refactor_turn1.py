def process_user_expression(expr: str) -> str:
    try:
        # Check for potentially unsafe elements
        unsafe_keywords = ['import', '__', 'exec', 'eval', 'compile', 'open', 'file', 'input', 'raw_input', 'execfile', 'globals', 'locals', 'vars', 'dir', 'help', 'breakpoint', 'quit', 'exit']
        
        # Convert to lowercase for case-insensitive checking
        expr_lower = expr.lower()
        
        # Check for unsafe keywords
        for keyword in unsafe_keywords:
            if keyword in expr_lower:
                return 'Processing Failed!'
        
        # Check for potentially dangerous characters or patterns
        if any(char in expr for char in [';', '\\', '`']):
            return 'Processing Failed!'
        
        # Create a restricted namespace with only safe built-ins
        safe_dict = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'pow': pow,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'all': all,
            'any': any,
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Use compile to check syntax before evaluation
        try:
            compiled = compile(expr, '<string>', 'eval')
        except SyntaxError:
            return 'Processing Failed!'
        
        # Evaluate the expression with restricted namespace
        result = eval(compiled, {"__builtins__": {}}, safe_dict)
        
        # Convert result to string
        return str(result)
        
    except Exception:
        return 'Processing Failed!'
