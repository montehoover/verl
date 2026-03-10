import ast

def run_python_snippet(snippet_string: str):
    """
    Executes a Python code snippet safely, ensuring only benign actions are carried out.
    
    Args:
        snippet_string: str - The user-provided Python code to be executed
        
    Returns:
        The resulting value if the code produces any output, or None otherwise
        
    Raises:
        ValueError: If the snippet attempts forbidden operations or contains invalid code
    """
    # Parse the code to check for forbidden operations
    try:
        tree = ast.parse(snippet_string)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    
    # Define forbidden node types and names
    forbidden_nodes = (
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
        ast.AsyncFor,
        ast.AsyncFunctionDef,
    )
    
    forbidden_names = {
        '__import__', 'eval', 'exec', 'compile', 'open', 'file', 'input',
        'raw_input', 'execfile', 'reload', 'vars', 'locals', 'globals',
        'dir', 'help', 'type', 'getattr', 'setattr', 'delattr', 'hasattr',
        'callable', 'classmethod', 'staticmethod', 'property', 'super',
        'isinstance', 'issubclass', 'print', 'exit', 'quit'
    }
    
    # Check for forbidden operations
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"Forbidden operation: {node.__class__.__name__}")
        
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise ValueError(f"Forbidden function: {node.id}")
        
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                raise ValueError(f"Forbidden function call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in ['__getattr__', '__setattr__', '__delattr__', 
                                     '__getattribute__', '__class__', '__dict__',
                                     '__module__', '__bases__', '__subclasses__']:
                    raise ValueError(f"Forbidden attribute access: {node.func.attr}")
    
    # Create a restricted execution environment
    safe_builtins = {
        'abs': abs,
        'all': all,
        'any': any,
        'bin': bin,
        'bool': bool,
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
        'range': range,
        'reversed': reversed,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'zip': zip,
        'True': True,
        'False': False,
        'None': None,
    }
    
    safe_globals = {'__builtins__': safe_builtins}
    safe_locals = {}
    
    # Execute the code in the restricted environment
    try:
        # Compile and execute the code
        compiled_code = compile(snippet_string, '<string>', 'exec')
        exec(compiled_code, safe_globals, safe_locals)
        
        # Try to evaluate the last expression to get a return value
        try:
            # Parse as single expression
            expr_tree = ast.parse(snippet_string.strip(), mode='eval')
            result = eval(compile(expr_tree, '<string>', 'eval'), safe_globals, safe_locals)
            return result
        except:
            # If it's not a single expression, check if any variables were created
            if safe_locals:
                # Return the last assigned value if any
                return list(safe_locals.values())[-1]
            return None
            
    except Exception as e:
        raise ValueError(f"Error executing snippet: {e}")
