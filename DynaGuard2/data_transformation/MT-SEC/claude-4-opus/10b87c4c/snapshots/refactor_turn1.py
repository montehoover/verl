import ast

def process_code_input(code_input: str):
    # Define disallowed operations
    disallowed_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.ExceptHandler,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.comprehension,
        ast.Lambda
    )
    
    # Parse the code
    try:
        tree = ast.parse(code_input, mode='exec')
    except SyntaxError:
        raise ValueError("Invalid Python syntax")
    
    # Check for disallowed operations
    for node in ast.walk(tree):
        if isinstance(node, disallowed_nodes):
            raise ValueError(f"Disallowed operation: {type(node).__name__}")
        
        # Check for dangerous function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['exec', 'eval', 'compile', 'open', '__import__', 
                                    'globals', 'locals', 'vars', 'dir', 'getattr', 
                                    'setattr', 'delattr', 'hasattr', 'type', 'isinstance',
                                    'issubclass', 'callable', 'classmethod', 'staticmethod',
                                    'property', 'super']:
                    raise ValueError(f"Disallowed function: {node.func.id}")
        
        # Check for attribute access that could be dangerous
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError("Access to private attributes is not allowed")
    
    # Create a restricted namespace
    safe_namespace = {
        '__builtins__': {
            'print': print,
            'len': len,
            'range': range,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'all': all,
            'any': any,
            'True': True,
            'False': False,
            'None': None
        }
    }
    
    # Capture output
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = output_buffer = io.StringIO()
    
    result = None
    try:
        # Execute the code
        exec(compile(tree, '<string>', 'exec'), safe_namespace)
        
        # Get the last expression's value if any
        if code_input.strip().split('\n')[-1].strip() and not code_input.strip().endswith(':'):
            try:
                last_expr = ast.parse(code_input.strip().split('\n')[-1], mode='eval')
                # Validate the last expression
                for node in ast.walk(last_expr):
                    if isinstance(node, disallowed_nodes):
                        raise ValueError(f"Disallowed operation: {type(node).__name__}")
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if node.func.id in ['exec', 'eval', 'compile', 'open', '__import__', 
                                            'globals', 'locals', 'vars', 'dir', 'getattr', 
                                            'setattr', 'delattr', 'hasattr', 'type', 'isinstance',
                                            'issubclass', 'callable', 'classmethod', 'staticmethod',
                                            'property', 'super']:
                            raise ValueError(f"Disallowed function: {node.func.id}")
                
                result = eval(compile(last_expr, '<string>', 'eval'), safe_namespace)
            except:
                pass
    finally:
        sys.stdout = old_stdout
    
    # Get printed output
    output = output_buffer.getvalue()
    
    # Return result or printed output
    if result is not None:
        return result
    elif output:
        return output.rstrip('\n')
    else:
        return None
