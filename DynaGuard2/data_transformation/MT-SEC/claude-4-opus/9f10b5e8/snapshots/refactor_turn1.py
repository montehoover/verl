def run_user_query(query: str) -> str:
    try:
        # Check for dangerous operations
        dangerous_keywords = [
            '__import__', 'exec', 'eval', 'compile', 'open',
            'file', 'input', 'raw_input', '__builtins__',
            'globals', 'locals', 'vars', 'dir', 'getattr',
            'setattr', 'delattr', 'hasattr', '__', 'import',
            'os', 'sys', 'subprocess', 'socket', 'requests',
            'urllib', 'pickle', 'shelve', 'marshal', 'tempfile',
            'shutil', 'pathlib', 'Path', 'breakpoint', 'help',
            'quit', 'exit', 'copyright', 'credits', 'license'
        ]
        
        query_lower = query.lower()
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                return 'Error!'
        
        # Evaluate the expression
        result = eval(query, {"__builtins__": {}}, {})
        return str(result)
        
    except:
        return 'Error!'
