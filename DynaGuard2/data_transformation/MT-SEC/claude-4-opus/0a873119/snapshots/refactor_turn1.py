import ast

def safe_run_script(script_code):
    """
    Execute a user-provided Python script, ensuring that only non-harmful operations are allowed.
    
    Args:
        script_code: a string containing the Python code submitted by the user
        
    Returns:
        The result of the executed script, or None if no result
        
    Raises:
        ValueError: if the script involves prohibited operations or contains invalid syntax
    """
    # Parse the script to check for prohibited operations
    try:
        tree = ast.parse(script_code)
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")
    
    # Define prohibited node types
    prohibited_nodes = {
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.Exec,
        ast.With,
        ast.AsyncWith,
        ast.Open,
    }
    
    # Define prohibited built-in functions
    prohibited_builtins = {
        'open', 'file', 'input', 'raw_input', 'compile', 'exec', 'eval',
        '__import__', 'globals', 'locals', 'vars', 'dir', 'help',
        'delattr', 'setattr', 'getattr', 'hasattr', 'reload',
        'type', 'isinstance', 'issubclass', 'super', 'property',
        'staticmethod', 'classmethod', 'callable', 'format',
        'memoryview', 'bytearray', 'bytes', 'frozenset',
        'enumerate', 'filter', 'map', 'zip', 'reversed',
        'sorted', 'iter', 'next', 'range', 'xrange',
        'breakpoint', 'exit', 'quit'
    }
    
    # Check for prohibited operations
    for node in ast.walk(tree):
        if type(node) in prohibited_nodes:
            raise ValueError(f"Prohibited operation: {type(node).__name__}")
        
        # Check for attribute access to prohibited modules
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ['os', 'sys', 'subprocess', 'socket', 'urllib', 'requests']:
                raise ValueError(f"Prohibited module access: {node.value.id}")
        
        # Check for calls to prohibited builtins
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in prohibited_builtins:
                raise ValueError(f"Prohibited function call: {node.func.id}")
    
    # Create a restricted environment
    safe_globals = {
        '__builtins__': {
            'print': print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'all': all,
            'any': any,
            'ord': ord,
            'chr': chr,
            'bin': bin,
            'hex': hex,
            'oct': oct,
            'pow': pow,
            'divmod': divmod,
            'True': True,
            'False': False,
            'None': None,
        }
    }
    
    safe_locals = {}
    
    # Execute the script
    try:
        exec(script_code, safe_globals, safe_locals)
        
        # Try to return the last expression's value
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(body=tree.body[-1].value)
            code = compile(last_expr, '<string>', 'eval')
            return eval(code, safe_globals, safe_locals)
        
        return None
        
    except Exception as e:
        raise ValueError(f"Execution error: {e}")
