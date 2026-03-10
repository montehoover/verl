import ast

def execute_python_code(snippet_code: str):
    """
    Executes a string of Python code safely, ensuring only benign actions are carried out.
    
    Args:
        snippet_code: str, the user-provided Python code to be executed
        
    Returns:
        Returns the resulting value if the code produces any output, or None otherwise
        
    Raises:
        ValueError: if the snippet attempts forbidden operations or contains invalid code
    """
    # List of forbidden built-in functions and modules
    forbidden_builtins = {
        'eval', 'exec', 'compile', '__import__', 'open', 'input',
        'file', 'execfile', 'reload', 'vars', 'globals', 'locals',
        'dir', 'help', 'quit', 'exit', 'license', 'copyright', 'credits'
    }
    
    forbidden_modules = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
        'importlib', 'pickle', 'shelve', 'tempfile', 'shutil',
        'pathlib', 'glob', 'io', 'ctypes', 'multiprocessing',
        'threading', 'concurrent', 'asyncio'
    }
    
    # Parse the code into an AST
    try:
        tree = ast.parse(snippet_code, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    
    # Check for forbidden operations in the AST
    for node in ast.walk(tree):
        # Check for imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in forbidden_modules:
                        raise ValueError(f"Import of module '{alias.name}' is not allowed")
            else:
                if node.module and node.module.split('.')[0] in forbidden_modules:
                    raise ValueError(f"Import from module '{node.module}' is not allowed")
        
        # Check for function calls to forbidden builtins
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_builtins:
                raise ValueError(f"Use of '{node.func.id}' is not allowed")
        
        # Check for attribute access that might be dangerous
        elif isinstance(node, ast.Attribute):
            # Check for __builtins__, __globals__, etc.
            if node.attr.startswith('__') and node.attr.endswith('__'):
                dangerous_attrs = {'__builtins__', '__globals__', '__locals__', 
                                 '__code__', '__class__', '__bases__', '__subclasses__'}
                if node.attr in dangerous_attrs:
                    raise ValueError(f"Access to '{node.attr}' is not allowed")
    
    # Create a restricted execution environment
    safe_builtins = {
        'abs': abs, 'all': all, 'any': any, 'ascii': ascii, 'bin': bin,
        'bool': bool, 'bytearray': bytearray, 'bytes': bytes, 'chr': chr,
        'complex': complex, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
        'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
        'hex': hex, 'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
        'iter': iter, 'len': len, 'list': list, 'map': map, 'max': max,
        'min': min, 'next': next, 'oct': oct, 'ord': ord, 'pow': pow,
        'print': print, 'range': range, 'repr': repr, 'reversed': reversed,
        'round': round, 'set': set, 'slice': slice, 'sorted': sorted,
        'str': str, 'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
        'None': None, 'True': True, 'False': False
    }
    
    # Create a clean namespace for execution
    namespace = {'__builtins__': safe_builtins}
    
    # Execute the code
    try:
        exec(compile(tree, '<string>', 'exec'), namespace)
    except Exception as e:
        raise ValueError(f"Error during execution: {e}")
    
    # Look for any returned value in the namespace
    # Remove built-in items to find user-defined variables
    result = None
    for key, value in namespace.items():
        if key != '__builtins__' and not key.startswith('__'):
            result = value
    
    return result
