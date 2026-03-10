import ast

def restricted_run_code(run_code):
    """
    Execute a Python script with restrictions to ensure safety.
    
    Args:
        run_code: A string containing the user's Python script
        
    Returns:
        The output of the script (if applicable), or None
        
    Raises:
        ValueError: If the script attempts prohibited actions or has syntax issues
    """
    # Define allowed node types for safe execution
    ALLOWED_NODE_TYPES = {
        # Literals and basic expressions
        ast.Module, ast.Expr, ast.Num, ast.Str, ast.Bytes, ast.NameConstant,
        ast.Constant,  # For Python 3.8+
        
        # Basic operations
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.FloorDiv, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor,
        ast.BitAnd, ast.MatMult,
        
        # Comparison operators
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
        
        # Boolean operators
        ast.And, ast.Or, ast.Not,
        
        # Unary operators
        ast.Invert, ast.UAdd, ast.USub,
        
        # Variables and assignments
        ast.Name, ast.Store, ast.Load, ast.Del, ast.Assign,
        ast.AugAssign,
        
        # Collections
        ast.List, ast.Tuple, ast.Set, ast.Dict,
        
        # Basic control flow
        ast.If, ast.For, ast.While, ast.Break, ast.Continue,
        ast.Pass,
        
        # List/Dict comprehensions
        ast.ListComp, ast.DictComp, ast.SetComp,
        ast.comprehension,
        
        # Indexing and slicing
        ast.Subscript, ast.Index, ast.Slice,
        
        # Basic function calls (will be filtered)
        ast.Call,
        
        # Attributes (will be filtered)
        ast.Attribute,
    }
    
    # Define allowed built-in functions
    ALLOWED_BUILTINS = {
        'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter',
        'float', 'int', 'len', 'list', 'map', 'max', 'min', 'range',
        'reversed', 'round', 'set', 'sorted', 'str', 'sum', 'tuple',
        'zip', 'chr', 'ord', 'hex', 'oct', 'bin',
    }
    
    def validate_ast(node):
        """Recursively validate AST nodes for safety."""
        if type(node) not in ALLOWED_NODE_TYPES:
            raise ValueError(f"Prohibited operation: {type(node).__name__}")
        
        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in ALLOWED_BUILTINS:
                    raise ValueError(f"Prohibited function call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                # Allow basic string, list, dict methods
                allowed_methods = {
                    'append', 'extend', 'insert', 'remove', 'pop', 'clear',
                    'index', 'count', 'sort', 'reverse', 'copy',
                    'upper', 'lower', 'strip', 'split', 'join', 'replace',
                    'startswith', 'endswith', 'find', 'format',
                    'keys', 'values', 'items', 'get', 'update',
                }
                if node.func.attr not in allowed_methods:
                    raise ValueError(f"Prohibited method call: {node.func.attr}")
            else:
                raise ValueError("Complex function calls are not allowed")
        
        # Check imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed")
        
        # Check for class definitions
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError("Function and class definitions are not allowed")
        
        # Check for exec/eval
        if isinstance(node, ast.Name) and node.id in ('exec', 'eval', 'compile', '__import__'):
            raise ValueError(f"Prohibited built-in: {node.id}")
        
        # Recursively validate child nodes
        for child in ast.iter_child_nodes(node):
            validate_ast(child)
    
    try:
        # Parse the code into an AST
        tree = ast.parse(run_code, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}")
    
    # Validate the AST for safety
    validate_ast(tree)
    
    # Create a restricted namespace
    safe_namespace = {
        '__builtins__': {name: __builtins__[name] for name in ALLOWED_BUILTINS if name in __builtins__},
        'True': True,
        'False': False,
        'None': None,
    }
    
    # Execute the code in the restricted namespace
    try:
        exec(compile(tree, '<string>', 'exec'), safe_namespace)
    except Exception as e:
        raise ValueError(f"Execution error: {e}")
    
    # The function returns None as specified (output would typically be captured differently)
    return None
