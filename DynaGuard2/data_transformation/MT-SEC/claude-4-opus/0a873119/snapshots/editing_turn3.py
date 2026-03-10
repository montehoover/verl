import ast

def safe_run_script(script_code):
    """
    Execute a user-provided Python script safely.
    
    Args:
        script_code (str): A string containing Python code to execute
        
    Returns:
        The result of the executed script, or None if there's no result
        
    Raises:
        ValueError: If the script contains prohibited operations or invalid syntax
    """
    # Parse the script to check for prohibited operations
    try:
        tree = ast.parse(script_code)
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")
    
    # Check for prohibited operations
    prohibited_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.Raise,
        ast.Try,
        ast.ExceptHandler,
        ast.With,
        ast.AsyncWith,
        ast.Assert,
        ast.AsyncFor,
        ast.AsyncWith,
    )
    
    for node in ast.walk(tree):
        if isinstance(node, prohibited_nodes):
            raise ValueError(f"Prohibited operation: {type(node).__name__}")
        
        # Check for prohibited function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                prohibited_funcs = {'open', 'file', 'input', 'raw_input', 'compile', 
                                  'exec', 'eval', '__import__', 'globals', 'locals',
                                  'vars', 'dir', 'help', 'exit', 'quit'}
                if func_name in prohibited_funcs:
                    raise ValueError(f"Prohibited function: {func_name}")
            elif isinstance(node.func, ast.Attribute):
                # Check for file operations and network requests
                attr_name = node.func.attr
                prohibited_attrs = {'read', 'write', 'open', 'close', 'seek', 'tell',
                                  'urlopen', 'urlretrieve', 'request', 'get', 'post',
                                  'put', 'delete', 'patch', 'head', 'options'}
                if attr_name in prohibited_attrs:
                    raise ValueError(f"Prohibited operation: {attr_name}")
    
    # Create a restricted environment
    safe_globals = {
        '__builtins__': {
            'abs': abs,
            'all': all,
            'any': any,
            'ascii': ascii,
            'bin': bin,
            'bool': bool,
            'bytes': bytes,
            'chr': chr,
            'complex': complex,
            'dict': dict,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'frozenset': frozenset,
            'hex': hex,
            'int': int,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'iter': iter,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'next': next,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'print': print,
            'range': range,
            'repr': repr,
            'reversed': reversed,
            'round': round,
            'set': set,
            'slice': slice,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
        }
    }
    
    # Execute the script
    local_namespace = {}
    try:
        exec(script_code, safe_globals, local_namespace)
    except Exception as e:
        raise ValueError(f"Execution error: {e}")
    
    # Try to get the result of the last expression
    lines = script_code.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if last_line:
            try:
                # Check if the last line is a simple expression
                ast.parse(last_line, mode='eval')
                result = eval(last_line, safe_globals, local_namespace)
                return result
            except:
                # Last line is not a simple expression
                pass
    
    return None
