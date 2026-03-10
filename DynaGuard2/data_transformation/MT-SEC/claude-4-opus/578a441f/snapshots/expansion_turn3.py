import ast

def validate_python_script(script: str) -> bool:
    """
    Validates that a Python script contains only safe operations.
    Returns True if the script is safe, False otherwise.
    """
    # Define allowed node types for a safe subset of Python
    ALLOWED_NODES = {
        # Literals and basic expressions
        ast.Constant,  # Python 3.8+ for literals
        ast.Num,       # Numbers (legacy, pre-3.8)
        ast.Str,       # Strings (legacy, pre-3.8)
        ast.Bytes,     # Bytes (legacy, pre-3.8)
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.NameConstant,  # True, False, None (legacy, pre-3.8)
        
        # Variables and names
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Del,
        
        # Basic operations
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        
        # Operators
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.LShift,
        ast.RShift,
        ast.BitOr,
        ast.BitXor,
        ast.BitAnd,
        ast.MatMult,
        
        # Unary operators
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Invert,
        
        # Boolean operators
        ast.And,
        ast.Or,
        
        # Comparison operators
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        
        # Control flow (limited)
        ast.If,
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.Pass,
        
        # Comprehensions
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.comprehension,
        
        # Function definitions (but not calls)
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        
        # Basic statements
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Expr,
        
        # Subscripting and slicing
        ast.Subscript,
        ast.Index,
        ast.Slice,
        
        # Context expressions
        ast.expr_context,
        
        # Module and body
        ast.Module,
        ast.Interactive,
        ast.Expression,
    }
    
    # Explicitly forbidden operations
    FORBIDDEN_NODES = {
        # Import statements
        ast.Import,
        ast.ImportFrom,
        
        # Function/method calls (most dangerous)
        ast.Call,
        
        # Attribute access (can access dangerous methods)
        ast.Attribute,
        
        # Exec/eval related
        ast.Global,
        ast.Nonlocal,
        
        # Class definitions
        ast.ClassDef,
        
        # Advanced features
        ast.Lambda,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
        ast.AsyncFunctionDef,
        ast.AsyncFor,
        ast.AsyncWith,
        
        # Context managers
        ast.With,
        ast.withitem,
        
        # Exception handling
        ast.Try,
        ast.ExceptHandler,
        ast.Raise,
        ast.Assert,
        
        # Deletions (except simple del)
        ast.Delete,
        
        # Star expressions
        ast.Starred,
        
        # Format strings (can be dangerous)
        ast.JoinedStr,
        ast.FormattedValue,
    }
    
    try:
        # Parse the script into an AST
        tree = ast.parse(script, mode='exec')
    except SyntaxError:
        # Invalid Python syntax
        return False
    
    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        # Check if the node type is explicitly forbidden
        if type(node) in FORBIDDEN_NODES:
            return False
        
        # Check if the node type is not in the allowed list
        if type(node) not in ALLOWED_NODES:
            # Special handling for some nodes that might be version-specific
            node_type_name = type(node).__name__
            
            # Allow some nodes that might not be in our list but are safe
            if node_type_name in ['expr_context', 'boolop', 'operator', 
                                  'unaryop', 'cmpop', 'keyword']:
                continue
                
            return False
    
    # Additional validation: check for dangerous built-in names
    dangerous_names = {
        '__import__', 'eval', 'exec', 'compile', 'open', 'input',
        'help', 'dir', 'vars', 'locals', 'globals', 'getattr',
        'setattr', 'delattr', 'hasattr', 'type', 'isinstance',
        'issubclass', 'super', 'property', 'classmethod', 
        'staticmethod', 'callable', 'format', 'repr', 'ascii',
        'ord', 'chr', 'bin', 'hex', 'oct', 'id', 'hash',
        'breakpoint', 'exit', 'quit', 'credits', 'license',
        'print',  # Even print could be considered unsafe in some contexts
        '__build_class__', '__debug__', '__doc__', '__loader__',
        '__name__', '__package__', '__spec__', '__path__',
        '__file__', '__cached__', 'memoryview', 'bytearray',
        'bytes', 'filter', 'map', 'zip', 'enumerate', 'reversed',
        'iter', 'next', 'range', 'slice'
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in dangerous_names:
            return False
    
    # If we made it here, the script appears to be safe
    return True


def execute_safe_code(tree: ast.AST):
    """
    Executes a validated AST and returns the result of the last expression.
    Returns None if there is no expression result.
    Raises an exception for any execution issues.
    """
    # Create a restricted namespace with only safe built-ins
    safe_builtins = {
        # Basic types
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'tuple': tuple,
        'dict': dict,
        'set': set,
        
        # Constants
        'True': True,
        'False': False,
        'None': None,
        
        # Safe functions
        'len': len,
        'min': min,
        'max': max,
        'sum': sum,
        'abs': abs,
        'round': round,
        'all': all,
        'any': any,
        'sorted': sorted,
        
        # Math operations (without importing math)
        'pow': pow,
        'divmod': divmod,
    }
    
    # Create isolated namespace
    namespace = {'__builtins__': safe_builtins}
    
    # Track the last expression result
    last_expr_result = None
    
    # If it's a Module, we need to handle it specially to track expression results
    if isinstance(tree, ast.Module):
        # Create a modified version of the tree where Expr nodes store their results
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.Expr):
                # Create an assignment that captures the expression result
                temp_var = f'__expr_result_{i}__'
                assign_node = ast.Assign(
                    targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                    value=node.value
                )
                # Replace the Expr node with the assignment
                tree.body[i] = assign_node
                # Track this as a potential last expression
                last_expr_var = temp_var
        
        # Compile and execute the modified tree
        compiled_code = compile(tree, '<safe_code>', 'exec')
        exec(compiled_code, namespace)
        
        # Find the last expression result
        for i in range(len(tree.body) - 1, -1, -1):
            temp_var = f'__expr_result_{i}__'
            if temp_var in namespace:
                last_expr_result = namespace[temp_var]
                break
    
    # If it's an Expression, evaluate it directly
    elif isinstance(tree, ast.Expression):
        compiled_code = compile(tree, '<safe_code>', 'eval')
        last_expr_result = eval(compiled_code, namespace)
    
    # For Interactive mode
    elif isinstance(tree, ast.Interactive):
        # Similar to Module but for interactive mode
        for node in tree.body:
            if isinstance(node, ast.Expr):
                compiled_expr = compile(ast.Expression(body=node.value), '<safe_code>', 'eval')
                last_expr_result = eval(compiled_expr, namespace)
            else:
                compiled_stmt = compile(ast.Module(body=[node], type_ignores=[]), '<safe_code>', 'exec')
                exec(compiled_stmt, namespace)
    
    else:
        # For other AST types, compile and execute normally
        compiled_code = compile(tree, '<safe_code>', 'exec')
        exec(compiled_code, namespace)
    
    return last_expr_result


def run_user_script(user_script: str):
    """
    Validates and executes a user script in a safe environment.
    
    Args:
        user_script: String containing Python code to execute
        
    Returns:
        The result of the last expression in the script, or None
        
    Raises:
        ValueError: If the script contains disallowed operations or syntax errors
    """
    # First, validate the script
    if not validate_python_script(user_script):
        raise ValueError("Script contains disallowed operations or syntax errors")
    
    # Parse the script to get the AST
    try:
        tree = ast.parse(user_script, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Script contains syntax errors: {e}")
    
    # Execute the safe code and return the result
    try:
        return execute_safe_code(tree)
    except Exception as e:
        # Re-raise any execution errors as ValueError to maintain consistent interface
        raise ValueError(f"Script execution failed: {e}")
