import ast

def safe_run_script(script_code):
    """
    Execute a user-provided Python script safely, allowing only non-harmful operations.
    
    Args:
        script_code: a string containing the Python code submitted by the user
        
    Returns:
        The result of the executed script, or None if no result
        
    Raises:
        ValueError: if the script involves prohibited operations or contains invalid syntax
    """
    
    # Define prohibited AST node types
    prohibited_nodes = {
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.Exec,  # Only in Python 2, but included for completeness
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.Assert,
        ast.AsyncFor,
        ast.AsyncWith,
        ast.Yield,
        ast.YieldFrom,
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
    }
    
    # Define prohibited built-in functions
    prohibited_builtins = {
        'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input',
        '__import__', 'reload', 'dir', 'globals', 'locals', 'vars',
        'getattr', 'setattr', 'delattr', 'hasattr', 'type', 'isinstance',
        'issubclass', 'callable', 'classmethod', 'staticmethod', 'property',
        'super', 'object', 'help', 'exit', 'quit', 'credits', 'license',
        'copyright', 'breakpoint', 'memoryview', 'bytearray', 'bytes',
    }
    
    try:
        # Parse the script code
        tree = ast.parse(script_code, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")
    
    # Check for prohibited operations
    for node in ast.walk(tree):
        if type(node) in prohibited_nodes:
            raise ValueError(f"Prohibited operation: {type(node).__name__}")
        
        # Check for attribute access that might be dangerous
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError(f"Access to private attributes is not allowed: {node.attr}")
        
        # Check for prohibited function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in prohibited_builtins:
                raise ValueError(f"Prohibited function: {node.func.id}")
    
    # Create a restricted global environment
    safe_globals = {
        '__builtins__': {
            # Allow only safe built-in functions
            'abs': abs,
            'all': all,
            'any': any,
            'ascii': ascii,
            'bin': bin,
            'bool': bool,
            'chr': chr,
            'dict': dict,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'hex': hex,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
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
            'zip': zip,
            # Math operations
            'True': True,
            'False': False,
            'None': None,
        }
    }
    
    # Create a local namespace for execution
    safe_locals = {}
    
    try:
        # Execute the code in the restricted environment
        exec(compile(tree, '<user_script>', 'exec'), safe_globals, safe_locals)
        
        # Check if there's a result to return
        # Try to get the last expression's value if it exists
        if script_code.strip().split('\n')[-1].strip() and not script_code.strip().endswith(':'):
            try:
                # Try to evaluate the last line as an expression
                last_line = script_code.strip().split('\n')[-1].strip()
                if '=' not in last_line or last_line.count('=') == last_line.count('=='):
                    result = eval(last_line, safe_globals, safe_locals)
                    return result
            except:
                pass
        
        return None
        
    except Exception as e:
        raise ValueError(f"Error executing script: {e}")
