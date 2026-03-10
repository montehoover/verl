import ast

def _get_prohibited_nodes():
    """Return tuple of prohibited AST node types."""
    return (
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

def _get_prohibited_names():
    """Return set of prohibited function/variable names."""
    return {
        'eval', 'exec', 'compile', 'open', 'input', 'print',
        '__import__', 'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'type', 'isinstance', 'issubclass', 'callable',
        'classmethod', 'staticmethod', 'property',
        'super', 'object', 'exit', 'quit',
        'help', 'copyright', 'credits', 'license'
    }

def _create_safe_namespace():
    """Create a restricted namespace for script execution."""
    return {
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

def _parse_python_code(python_code):
    """Parse Python code and return AST tree."""
    try:
        return ast.parse(python_code, mode='exec')
    except SyntaxError:
        raise ValueError("Invalid syntax in the provided Python code")

def _validate_ast_tree(tree):
    """Validate AST tree for prohibited operations."""
    prohibited_nodes = _get_prohibited_nodes()
    prohibited_names = _get_prohibited_names()
    
    for node in ast.walk(tree):
        if isinstance(node, prohibited_nodes):
            raise ValueError(f"Prohibited operation: {node.__class__.__name__}")
        
        if isinstance(node, ast.Name) and node.id in prohibited_names:
            raise ValueError(f"Prohibited name: {node.id}")
        
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError(f"Access to private attributes is prohibited: {node.attr}")

def _execute_validated_code(tree, namespace):
    """Execute validated AST tree in the given namespace."""
    try:
        exec(compile(tree, '<string>', 'exec'), namespace)
    except Exception as e:
        raise ValueError(f"Error executing script: {str(e)}")

def _extract_last_expression_result(tree, namespace):
    """Extract and evaluate the last expression if present."""
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(body=tree.body[-1].value)
        try:
            return eval(compile(last_expr, '<string>', 'eval'), namespace)
        except:
            pass
    return None

def execute_python_script(python_code):
    # Parse the code
    tree = _parse_python_code(python_code)
    
    # Validate the AST tree
    _validate_ast_tree(tree)
    
    # Create a restricted namespace
    namespace = _create_safe_namespace()
    
    # Execute the code
    _execute_validated_code(tree, namespace)
    
    # Find and return the last expression value
    return _extract_last_expression_result(tree, namespace)
