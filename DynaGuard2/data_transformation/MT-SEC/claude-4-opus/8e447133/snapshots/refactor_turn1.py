def execute_user_input(user_code: str) -> str:
    # Define allowed names for safe execution
    allowed_names = {
        # Basic arithmetic operations
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'len': len,
        # Type conversions
        'int': int,
        'float': float,
        'str': str,
        # String methods are allowed on string objects
    }
    
    # Check for dangerous keywords and constructs
    dangerous_keywords = [
        'import', 'from', '__import__', 'exec', 'eval', 'compile',
        'open', 'file', 'input', 'raw_input', 'globals', 'locals',
        'vars', 'dir', 'delattr', 'setattr', 'getattr', 'hasattr',
        '__', 'lambda', 'def', 'class', 'for', 'while', 'with',
        'try', 'except', 'finally', 'raise', 'assert', 'del',
        'global', 'nonlocal', 'yield', 'async', 'await',
        'breakpoint', 'help', 'quit', 'exit', 'copyright', 'credits',
        'license'
    ]
    
    # Check for dangerous patterns in the code
    code_lower = user_code.lower()
    for keyword in dangerous_keywords:
        if keyword in code_lower:
            return 'Execution Blocked!'
    
    # Check for attribute access that could be dangerous
    if '.' in user_code:
        # Allow only basic string methods
        allowed_string_methods = {
            'upper', 'lower', 'strip', 'lstrip', 'rstrip',
            'replace', 'split', 'join', 'startswith', 'endswith',
            'find', 'count', 'isdigit', 'isalpha', 'isalnum',
            'title', 'capitalize', 'swapcase'
        }
        
        # Simple check for method calls
        import re
        method_calls = re.findall(r'\.(\w+)\s*\(', user_code)
        for method in method_calls:
            if method not in allowed_string_methods:
                return 'Execution Blocked!'
    
    try:
        # Compile the code to check for syntax errors
        compiled_code = compile(user_code, '<user_input>', 'eval')
        
        # Create a restricted global namespace
        restricted_globals = {'__builtins__': allowed_names}
        
        # Evaluate the expression
        result = eval(compiled_code, restricted_globals, {})
        
        # Convert result to string
        return str(result)
        
    except:
        # If compilation fails, try as a statement
        try:
            # For simple assignments or print statements
            compiled_code = compile(user_code, '<user_input>', 'exec')
            
            # Create a restricted namespace for execution
            restricted_globals = {'__builtins__': allowed_names}
            local_namespace = {}
            
            # Execute the code
            exec(compiled_code, restricted_globals, local_namespace)
            
            # If there's a result variable, return it
            if local_namespace:
                # Return the last assigned variable
                last_value = list(local_namespace.values())[-1]
                return str(last_value)
            else:
                return ''
                
        except:
            return 'Execution Blocked!'
