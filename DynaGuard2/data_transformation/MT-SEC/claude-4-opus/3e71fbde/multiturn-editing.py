import ast

def run_code_snippet(code_snippet):
    """
    Execute a Python code snippet with strict safety rules.
    
    Args:
        code_snippet (str): A string containing Python code to execute
    
    Returns:
        any: The result of the execution, or None if there's no output
        
    Raises:
        ValueError: If the code tries to perform disallowed operations
    """
    # Parse the code to check for disallowed operations
    try:
        tree = ast.parse(code_snippet)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    
    # Check for disallowed operations
    for node in ast.walk(tree):
        # Disallow imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed")
        
        # Disallow file operations
        if isinstance(node, ast.Name) and node.id in ['open', 'file']:
            raise ValueError("File operations are not allowed")
        
        # Disallow exec/eval
        if isinstance(node, ast.Name) and node.id in ['exec', 'eval', 'compile', '__import__']:
            raise ValueError("Dynamic code execution is not allowed")
        
        # Disallow access to __builtins__ or other dangerous attributes
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            raise ValueError("Access to special attributes is not allowed")
    
    # Create a restricted namespace
    safe_builtins = {
        'abs': abs, 'all': all, 'any': any, 'bool': bool,
        'dict': dict, 'enumerate': enumerate, 'float': float,
        'int': int, 'len': len, 'list': list, 'max': max,
        'min': min, 'print': print, 'range': range, 'round': round,
        'set': set, 'sorted': sorted, 'str': str, 'sum': sum,
        'tuple': tuple, 'type': type, 'zip': zip,
        'True': True, 'False': False, 'None': None
    }
    
    namespace = {'__builtins__': safe_builtins}
    
    # Execute the code
    try:
        exec(compile(tree, '<string>', 'exec'), namespace)
    except Exception as e:
        raise ValueError(f"Error during execution: {e}")
    
    # Try to get the result of the last expression
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        try:
            result = eval(compile(ast.Expression(tree.body[-1].value), '<string>', 'eval'), namespace)
            return result
        except:
            pass
    
    # Check if any variables were created
    for key, value in namespace.items():
        if key != '__builtins__' and not key.startswith('_'):
            return value
    
    return None
