import ast

def restricted_run_code(run_code):
    """Execute a Python script with restricted operations.
    
    Args:
        run_code: A string containing Python code to execute
        
    Returns:
        The output of the script or None if there's no output
        
    Raises:
        ValueError: If the script contains prohibited actions or has syntax issues
    """
    # Define allowed node types for safe execution
    allowed_nodes = {
        ast.Module, ast.Expr, ast.Load, ast.Store, ast.Del,
        ast.Assign, ast.AugAssign, ast.AnnAssign,
        ast.For, ast.While, ast.If, ast.With, ast.withitem,
        ast.Raise, ast.Try, ast.ExceptHandler, ast.Assert,
        ast.Global, ast.Nonlocal, ast.Pass, ast.Break, ast.Continue,
        ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Lambda,
        ast.IfExp, ast.Dict, ast.Set, ast.ListComp, ast.SetComp,
        ast.DictComp, ast.GeneratorExp, ast.Yield, ast.YieldFrom,
        ast.Compare, ast.Call, ast.Constant, ast.Attribute,
        ast.Subscript, ast.Starred, ast.Name, ast.List, ast.Tuple,
        ast.Slice, ast.ExtSlice, ast.Index,
        ast.And, ast.Or, ast.Add, ast.Sub, ast.Mult, ast.MatMult,
        ast.Div, ast.Mod, ast.Pow, ast.LShift, ast.RShift,
        ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv,
        ast.Invert, ast.Not, ast.UAdd, ast.USub,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
        ast.comprehension, ast.ExceptHandler, ast.arguments,
        ast.arg, ast.keyword, ast.alias
    }
    
    # Parse the code
    try:
        tree = ast.parse(run_code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}")
    
    # Check for prohibited nodes
    for node in ast.walk(tree):
        if type(node) not in allowed_nodes:
            raise ValueError(f"Prohibited operation: {type(node).__name__}")
        
        # Additional checks for specific nodes
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError("Access to names starting with '_' is prohibited")
            
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError("Access to attributes starting with '_' is prohibited")
    
    # Create a restricted environment
    safe_builtins = {
        'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
        'bin': bin, 'bool': bool, 'bytearray': bytearray,
        'bytes': bytes, 'chr': chr, 'complex': complex,
        'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
        'filter': filter, 'float': float, 'format': format,
        'frozenset': frozenset, 'hex': hex, 'int': int,
        'isinstance': isinstance, 'issubclass': issubclass,
        'iter': iter, 'len': len, 'list': list, 'map': map,
        'max': max, 'min': min, 'next': next, 'oct': oct,
        'ord': ord, 'pow': pow, 'print': print, 'range': range,
        'repr': repr, 'reversed': reversed, 'round': round,
        'set': set, 'slice': slice, 'sorted': sorted, 'str': str,
        'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip
    }
    
    # Execute the code in a restricted environment
    restricted_globals = {'__builtins__': safe_builtins}
    restricted_locals = {}
    
    # Compile and execute
    try:
        compiled = compile(tree, '<string>', 'exec')
        exec(compiled, restricted_globals, restricted_locals)
    except Exception as e:
        raise ValueError(f"Execution error: {e}")
    
    # Return the last expression value if any
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        try:
            last_expr = ast.Expression(tree.body[-1].value)
            compiled_expr = compile(last_expr, '<string>', 'eval')
            return eval(compiled_expr, restricted_globals, restricted_locals)
        except:
            pass
    
    return None


def execute_operation(operation):
    """Execute a simple arithmetic operation given as a string.
    
    Args:
        operation: A string containing a simple arithmetic operation (e.g., '5 + 3')
        
    Returns:
        The result of the arithmetic operation
    """
    # Create a local namespace for execution
    local_namespace = {}
    
    # Split the operation into individual statements
    statements = operation.split(';')
    
    # Execute each statement
    result = None
    for statement in statements:
        statement = statement.strip()
        if statement:
            result = eval(statement, {"__builtins__": {}}, local_namespace)
            # If it's an assignment, update the namespace
            if '=' in statement and not any(op in statement for op in ['==', '!=', '<=', '>=']):
                exec(statement, {"__builtins__": {}}, local_namespace)
    
    return result
