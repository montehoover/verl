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
