import ast

def execute_python_script(python_code):
    """
    Execute a user-provided Python script, ensuring that only non-harmful operations are allowed.
    
    Args:
        python_code: a string containing the Python code submitted by the user
        
    Returns:
        The result of the executed script, or None if no result
        
    Raises:
        ValueError: if the script involves prohibited operations or contains invalid syntax
    """
    # Define prohibited AST node types that could be harmful
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
        ast.Raise,
        ast.Try,
        ast.Assert,
        ast.Yield,
        ast.YieldFrom,
        ast.Lambda,
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.comprehension,
    }
    
    # Define prohibited built-in functions and attributes
    prohibited_names = {
        'open', 'file', 'input', 'raw_input', 'compile', 'exec', 'eval',
        'execfile', 'globals', 'locals', 'vars', 'dir', 'help',
        '__import__', 'reload', 'delattr', 'getattr', 'hasattr', 'setattr',
        'type', 'isinstance', 'issubclass', 'super', 'property',
        'staticmethod', 'classmethod', 'callable', 'format', 'repr',
        '__name__', '__file__', '__builtins__', '__loader__', '__package__',
        '__spec__', '__cached__', '__doc__', '__module__', '__dict__',
        '__weakref__', '__annotations__', '__class__', '__bases__',
        '__subclasses__', '__init__', '__new__', '__del__', '__call__',
        '__getattribute__', '__setattr__', '__delattr__', '__getitem__',
        '__setitem__', '__delitem__', '__iter__', '__next__', '__reversed__',
        '__contains__', '__enter__', '__exit__', 'print',  # print could leak info
        'breakpoint', 'exit', 'quit', 'credits', 'license', 'copyright'
    }
    
    try:
        # Parse the code into an AST
        tree = ast.parse(python_code, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {str(e)}")
    
    # Check for prohibited nodes
    for node in ast.walk(tree):
        if type(node) in prohibited_nodes:
            raise ValueError(f"Prohibited operation: {type(node).__name__}")
        
        # Check for attribute access that could be dangerous
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError(f"Access to private attributes is not allowed: {node.attr}")
        
        # Check for name access to prohibited built-ins
        if isinstance(node, ast.Name):
            if node.id in prohibited_names:
                raise ValueError(f"Use of prohibited name: {node.id}")
        
        # Check for function calls to prohibited built-ins
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in prohibited_names:
                raise ValueError(f"Call to prohibited function: {node.func.id}")
    
    # Create a restricted namespace for execution
    safe_namespace = {
        '__builtins__': {
            # Only allow safe built-in functions
            'abs': abs,
            'all': all,
            'any': any,
            'bin': bin,
            'bool': bool,
            'chr': chr,
            'dict': dict,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
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
            'range': range,
            'reversed': reversed,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
            'True': True,
            'False': False,
            'None': None,
        }
    }
    
    # Execute the code
    try:
        exec(compile(tree, '<user_script>', 'exec'), safe_namespace)
    except Exception as e:
        raise ValueError(f"Execution error: {str(e)}")
    
    # Return None as specified (no result for exec)
    return None
