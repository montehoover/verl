import ast

def execute_user_script(submitted_script):
    """
    Safely executes a Python script provided by the user while restricting it to only safe operations.
    
    Args:
        submitted_script: string, the script that the user submits for execution
        
    Returns:
        The output of the given script, if any, otherwise None
        
    Raises:
        ValueError: if the user script involves forbidden actions or contains syntax errors
    """
    
    # Define forbidden AST node types that could be dangerous
    forbidden_nodes = {
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.With,
        ast.AsyncWith,
        ast.Raise,
        ast.Try,
        ast.Global,
        ast.Nonlocal,
        ast.Exec,  # Only in Python 2, but checking anyway
        ast.Yield,
        ast.YieldFrom,
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.comprehension,
    }
    
    # Define forbidden built-in functions
    forbidden_builtins = {
        'eval', 'exec', 'compile', 'open', 'input', '__import__',
        'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
        'delattr', 'hasattr', 'type', 'id', 'help', 'reload',
        'breakpoint', 'exit', 'quit', 'copyright', 'credits', 'license'
    }
    
    try:
        # Parse the script into an AST
        tree = ast.parse(submitted_script)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in script: {e}")
    
    # Check for forbidden nodes
    for node in ast.walk(tree):
        if type(node) in forbidden_nodes:
            raise ValueError(f"Forbidden operation: {type(node).__name__}")
        
        # Check for attribute access that could be dangerous
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError("Access to private attributes is forbidden")
        
        # Check for calls to forbidden built-ins
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_builtins:
                raise ValueError(f"Forbidden function call: {node.func.id}")
    
    # Create a restricted global namespace
    safe_globals = {
        '__builtins__': {
            # Only allow safe built-in functions
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'pow': pow,
            'range': range,
            'reversed': reversed,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
            'print': print,
        }
    }
    
    # Execute the script in the restricted environment
    try:
        # Compile and execute the code
        compiled_code = compile(tree, '<user_script>', 'exec')
        
        # Capture print output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            exec(compiled_code, safe_globals)
            output = captured_output.getvalue()
            return output if output else None
        finally:
            sys.stdout = old_stdout
            
    except Exception as e:
        raise ValueError(f"Error during script execution: {e}")
