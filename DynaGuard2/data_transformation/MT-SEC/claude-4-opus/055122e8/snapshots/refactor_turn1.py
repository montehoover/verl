import ast

def execute_python_script(python_code):
    # Define prohibited operations
    prohibited_nodes = (
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
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Await,
        ast.AsyncFor,
    )
    
    prohibited_names = {
        'eval', 'exec', 'compile', 'open', 'input', 'print',
        '__import__', 'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'type', 'isinstance', 'issubclass', 'callable',
        'classmethod', 'staticmethod', 'property',
        'super', 'object', 'exit', 'quit',
        'help', 'copyright', 'credits', 'license'
    }
    
    # Parse the code
    try:
        tree = ast.parse(python_code, mode='exec')
    except SyntaxError:
        raise ValueError("Invalid syntax in the provided Python code")
    
    # Check for prohibited operations
    for node in ast.walk(tree):
        if isinstance(node, prohibited_nodes):
            raise ValueError(f"Prohibited operation: {node.__class__.__name__}")
        
        if isinstance(node, ast.Name) and node.id in prohibited_names:
            raise ValueError(f"Prohibited name: {node.id}")
        
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError(f"Access to private attributes is prohibited: {node.attr}")
    
    # Create a restricted namespace
    namespace = {
        '__builtins__': {
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
            'ord': ord,
            'chr': chr,
            'bin': bin,
            'hex': hex,
            'oct': oct,
            'pow': pow,
            'divmod': divmod,
            'None': None,
            'True': True,
            'False': False,
        }
    }
    
    # Execute the code
    try:
        exec(compile(tree, '<string>', 'exec'), namespace)
    except Exception as e:
        raise ValueError(f"Error executing script: {str(e)}")
    
    # Find the last expression to return its value
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(body=tree.body[-1].value)
        try:
            result = eval(compile(last_expr, '<string>', 'eval'), namespace)
            return result
        except:
            pass
    
    return None
