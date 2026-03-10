import ast

# Define prohibited node types
PROHIBITED_NODES = {
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
PROHIBITED_BUILTINS = {
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

# Define prohibited module names
PROHIBITED_MODULES = {'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests'}

# Define safe builtins
SAFE_BUILTINS = {
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


def parse_script(script_code):
    """
    Parse the script code and return the AST tree.
    
    Args:
        script_code: a string containing the Python code
        
    Returns:
        ast.Module: The parsed AST tree
        
    Raises:
        ValueError: if the script contains invalid syntax
    """
    try:
        return ast.parse(script_code)
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")


def validate_ast_node(node):
    """
    Validate a single AST node for prohibited operations.
    
    Args:
        node: An AST node to validate
        
    Raises:
        ValueError: if the node represents a prohibited operation
    """
    if type(node) in PROHIBITED_NODES:
        raise ValueError(f"Prohibited operation: {type(node).__name__}")
    
    # Check for attribute access to prohibited modules
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in PROHIBITED_MODULES:
            raise ValueError(f"Prohibited module access: {node.value.id}")
    
    # Check for calls to prohibited builtins
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in PROHIBITED_BUILTINS:
            raise ValueError(f"Prohibited function call: {node.func.id}")


def validate_script(tree):
    """
    Validate the entire AST tree for prohibited operations.
    
    Args:
        tree: The AST tree to validate
        
    Raises:
        ValueError: if the tree contains prohibited operations
    """
    for node in ast.walk(tree):
        validate_ast_node(node)


def create_safe_environment():
    """
    Create a safe execution environment with restricted globals and locals.
    
    Returns:
        tuple: (safe_globals, safe_locals) dictionaries
    """
    safe_globals = {
        '__builtins__': SAFE_BUILTINS
    }
    safe_locals = {}
    return safe_globals, safe_locals


def execute_script(script_code, tree, safe_globals, safe_locals):
    """
    Execute the script in the safe environment and return the result.
    
    Args:
        script_code: The script code as a string
        tree: The parsed AST tree
        safe_globals: The safe globals dictionary
        safe_locals: The safe locals dictionary
        
    Returns:
        The result of the executed script, or None if no result
        
    Raises:
        ValueError: if execution fails
    """
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
    # Parse the script
    tree = parse_script(script_code)
    
    # Validate the script
    validate_script(tree)
    
    # Create safe environment
    safe_globals, safe_locals = create_safe_environment()
    
    # Execute the script
    return execute_script(script_code, tree, safe_globals, safe_locals)
