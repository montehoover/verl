import ast

def run_user_script(user_script: str):
    """
    Execute a Python script supplied by the user with restricted operations.
    
    Args:
        user_script: str, the Python script provided by the user
        
    Returns:
        The result of the script if any, or None
        
    Raises:
        ValueError: if the script contains disallowed operations or syntax errors
    """
    # List of allowed node types for safe execution
    allowed_nodes = {
        ast.Module, ast.Expr, ast.Load, ast.Store, ast.Del,
        ast.Assign, ast.AugAssign, ast.AnnAssign,
        ast.For, ast.While, ast.If, ast.With, ast.AsyncWith,
        ast.Raise, ast.Try, ast.Assert, ast.Pass, ast.Break, ast.Continue,
        ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Lambda, ast.IfExp,
        ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp,
        ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom,
        ast.Compare, ast.Call, ast.Constant, ast.FormattedValue,
        ast.JoinedStr, ast.Attribute, ast.Subscript, ast.Starred,
        ast.Name, ast.List, ast.Tuple, ast.Slice,
        ast.And, ast.Or, ast.Add, ast.Sub, ast.Mult, ast.MatMult,
        ast.Div, ast.Mod, ast.Pow, ast.LShift, ast.RShift,
        ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv,
        ast.Invert, ast.Not, ast.UAdd, ast.USub,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
        ast.comprehension, ast.ExceptHandler, ast.arguments,
        ast.arg, ast.keyword, ast.alias, ast.withitem,
        ast.Return, ast.Delete, ast.Index, ast.ExtSlice
    }
    
    # List of allowed built-in functions
    allowed_builtins = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'chr', 'complex', 'dict', 'divmod', 'enumerate', 'filter', 'float',
        'format', 'frozenset', 'hash', 'hex', 'int', 'isinstance', 'issubclass',
        'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'oct', 'ord',
        'pow', 'print', 'range', 'repr', 'reversed', 'round', 'set',
        'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
    }
    
    try:
        # Parse the script into an AST
        tree = ast.parse(user_script, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Syntax error in script: {e}")
    
    # Check for disallowed operations
    for node in ast.walk(tree):
        # Check if node type is allowed
        if type(node) not in allowed_nodes:
            # Special handling for Import and ImportFrom
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Import statements are not allowed")
            # Special handling for FunctionDef and ClassDef
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                raise ValueError("Function and class definitions are not allowed")
            # Special handling for Global and Nonlocal
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                raise ValueError("Global and nonlocal statements are not allowed")
            else:
                raise ValueError(f"Disallowed operation: {type(node).__name__}")
        
        # Check for disallowed built-in functions
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in allowed_builtins and node.func.id in __builtins__:
                if node.func.id in ('eval', 'exec', 'compile', '__import__', 'open', 
                                   'input', 'breakpoint', 'help', 'exit', 'quit'):
                    raise ValueError(f"Disallowed built-in function: {node.func.id}")
        
        # Check for attribute access on modules or dangerous objects
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError("Access to private attributes is not allowed")
    
    # Create a restricted environment
    restricted_globals = {
        '__builtins__': {name: __builtins__[name] for name in allowed_builtins if name in __builtins__}
    }
    
    # Execute the script
    try:
        exec(tree, restricted_globals)
        # Check if there's a result to return
        if user_script.strip().split('\n')[-1].strip() and not user_script.strip().endswith(':'):
            # Try to evaluate the last line as an expression
            try:
                last_line = user_script.strip().split('\n')[-1].strip()
                result = eval(last_line, restricted_globals)
                return result
            except:
                return None
        return None
    except Exception as e:
        raise ValueError(f"Error executing script: {e}")
