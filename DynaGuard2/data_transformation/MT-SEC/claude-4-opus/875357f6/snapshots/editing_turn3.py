import ast

def manage_script(user_script: str) -> str:
    # Check for length limit
    if len(user_script) > 100:
        raise ValueError("Script exceeds maximum length of 100 characters")
    
    # Check for suspicious keywords
    suspicious_keywords = ['exec', 'eval']
    for keyword in suspicious_keywords:
        if keyword in user_script:
            raise ValueError(f"Script contains suspicious keyword: {keyword}")
    
    # Try to compile the script to check if it's malformed
    try:
        compile(user_script, '<string>', 'exec')
    except SyntaxError:
        raise ValueError("Script is malformed")
    
    # Append the custom comment
    processed_script = user_script + "\n# Processed"
    
    return processed_script

def execute_user_script(submitted_script: str):
    # Define forbidden node types
    forbidden_nodes = (
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
        ast.Assert,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.comprehension,
        ast.Lambda,
        ast.Call,
    )
    
    # Parse the script
    try:
        tree = ast.parse(submitted_script, mode='exec')
    except SyntaxError:
        raise ValueError("Script contains syntax errors")
    
    # Check for forbidden operations
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"Forbidden operation: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in ['exec', 'eval', '__import__', 'open', 'compile']:
            raise ValueError(f"Forbidden function: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr in ['__class__', '__bases__', '__subclasses__', '__globals__', '__locals__', '__dict__']:
                raise ValueError(f"Forbidden attribute: {node.attr}")
    
    # Create a restricted namespace
    safe_globals = {
        '__builtins__': {
            'None': None,
            'True': True,
            'False': False,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'len': len,
            'range': range,
            'print': print,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
        }
    }
    
    # Execute the script
    output = []
    original_print = safe_globals['__builtins__']['print']
    
    def capture_print(*args, **kwargs):
        output.append(' '.join(str(arg) for arg in args))
    
    safe_globals['__builtins__']['print'] = capture_print
    
    try:
        exec(submitted_script, safe_globals)
    except Exception as e:
        raise ValueError(f"Execution error: {str(e)}")
    
    # Return output or None
    if output:
        return '\n'.join(output)
    else:
        return None
