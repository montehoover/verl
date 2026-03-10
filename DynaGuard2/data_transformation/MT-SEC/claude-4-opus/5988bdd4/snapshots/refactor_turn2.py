import ast

def _parse_and_validate_snippet(snippet_string: str) -> ast.AST:
    """
    Parse Python code and validate it doesn't contain forbidden operations.
    
    Args:
        snippet_string: The Python code to parse and validate
        
    Returns:
        The parsed AST tree
        
    Raises:
        ValueError: If the code contains invalid syntax or forbidden operations
    """
    # Parse the code
    try:
        tree = ast.parse(snippet_string)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    
    # Define forbidden node types
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
    
    # Define forbidden function names
    forbidden_names = {
        '__import__', 'eval', 'exec', 'compile', 'open', 'file', 'input',
        'raw_input', 'execfile', 'reload', 'vars', 'locals', 'globals',
        'dir', 'help', 'type', 'getattr', 'setattr', 'delattr', 'hasattr',
        'callable', 'classmethod', 'staticmethod', 'property', 'super',
        'isinstance', 'issubclass', 'print', 'exit', 'quit'
    }
    
    # Define forbidden attributes
    forbidden_attrs = {
        '__getattr__', '__setattr__', '__delattr__', '__getattribute__',
        '__class__', '__dict__', '__module__', '__bases__', '__subclasses__'
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
            elif isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attrs:
                raise ValueError(f"Forbidden attribute access: {node.func.attr}")
    
    return tree


def _create_safe_environment() -> dict:
    """
    Create a safe execution environment with limited builtins.
    
    Returns:
        A dictionary containing safe global variables
    """
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
    
    return {'__builtins__': safe_builtins}


def _execute_snippet(snippet_string: str, safe_globals: dict) -> tuple:
    """
    Execute a Python snippet in a safe environment.
    
    Args:
        snippet_string: The Python code to execute
        safe_globals: The safe global environment dictionary
        
    Returns:
        A tuple of (result, locals_dict) where result is the output value
        and locals_dict contains any variables created during execution
        
    Raises:
        ValueError: If execution fails
    """
    safe_locals = {}
    
    try:
        # Compile and execute the code
        compiled_code = compile(snippet_string, '<string>', 'exec')
        exec(compiled_code, safe_globals, safe_locals)
        
        # Try to evaluate as a single expression to get return value
        try:
            expr_tree = ast.parse(snippet_string.strip(), mode='eval')
            result = eval(compile(expr_tree, '<string>', 'eval'), safe_globals, safe_locals)
            return result, safe_locals
        except:
            return None, safe_locals
            
    except Exception as e:
        raise ValueError(f"Error executing snippet: {e}")


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
    # Parse and validate the snippet
    _parse_and_validate_snippet(snippet_string)
    
    # Create safe execution environment
    safe_globals = _create_safe_environment()
    
    # Execute the snippet
    result, safe_locals = _execute_snippet(snippet_string, safe_globals)
    
    # Return the result or last assigned value
    if result is not None:
        return result
    elif safe_locals:
        return list(safe_locals.values())[-1]
    else:
        return None
