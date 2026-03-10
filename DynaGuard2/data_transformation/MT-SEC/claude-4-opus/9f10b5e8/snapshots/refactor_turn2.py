def _is_query_safe(query: str) -> bool:
    """Check if the query contains any dangerous keywords."""
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
            return False
    return True


def _evaluate_query(query: str) -> str:
    """Evaluate the query and return the result as a string."""
    try:
        result = eval(query, {"__builtins__": {}}, {})
        return str(result)
    except:
        raise


def run_user_query(query: str) -> str:
    try:
        if not _is_query_safe(query):
            return 'Error!'
        
        return _evaluate_query(query)
        
    except:
        return 'Error!'
