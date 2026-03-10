import ast
import io
import sys

def _get_allowed_nodes():
    """Returns the set of allowed AST node types for safe execution."""
    return {
        ast.Module, ast.Expr, ast.Load, ast.Store, ast.Del,
        ast.Assign, ast.AugAssign, ast.AnnAssign,
        ast.For, ast.While, ast.If, ast.With, ast.withitem,
        ast.Break, ast.Continue, ast.Pass,
        ast.BoolOp, ast.BinOp, ast.UnaryOp,
        ast.Lambda, ast.IfExp,
        ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp,
        ast.GeneratorExp,
        ast.Yield, ast.YieldFrom,
        ast.Compare, ast.Call, ast.Constant, ast.Attribute,
        ast.Subscript, ast.Starred, ast.Name, ast.List, ast.Tuple,
        ast.Slice,
        ast.And, ast.Or,
        ast.Add, ast.Sub, ast.Mult, ast.MatMult, ast.Div, ast.Mod,
        ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor,
        ast.BitAnd, ast.FloorDiv,
        ast.Invert, ast.Not, ast.UAdd, ast.USub,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
        ast.comprehension, ast.ExceptHandler, ast.arguments,
        ast.arg, ast.keyword, ast.alias
    }

def _get_forbidden_builtins():
    """Returns the set of forbidden built-in function names."""
    return {
        'exec', 'eval', 'compile', '__import__', 'open',
        'input', 'help', 'dir', 'vars', 'locals', 'globals',
        'delattr', 'setattr', 'getattr', 'hasattr',
        'breakpoint', 'exit', 'quit'
    }

def _get_restricted_globals():
    """Returns the restricted global environment for safe script execution."""
    return {
        '__builtins__': {
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'all': all,
            'any': any,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'frozenset': frozenset,
            'type': type,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'ord': ord,
            'chr': chr,
            'bin': bin,
            'hex': hex,
            'oct': oct,
            'divmod': divmod,
            'pow': pow,
            'slice': slice,
            'None': None,
            'True': True,
            'False': False
        }
    }

def _validate_ast_node(node, allowed_nodes, forbidden_builtins):
    """
    Validates a single AST node for security.
    
    Args:
        node: The AST node to validate
        allowed_nodes: Set of allowed node types
        forbidden_builtins: Set of forbidden built-in function names
        
    Raises:
        ValueError: If the node represents a forbidden operation
    """
    if type(node) not in allowed_nodes:
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            raise ValueError("Import statements are not allowed")
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            raise ValueError("Function definitions are not allowed")
        elif isinstance(node, ast.ClassDef):
            raise ValueError("Class definitions are not allowed")
        elif isinstance(node, ast.Try):
            raise ValueError("Try/except blocks are not allowed")
        elif isinstance(node, ast.Raise):
            raise ValueError("Raise statements are not allowed")
        elif isinstance(node, ast.Assert):
            raise ValueError("Assert statements are not allowed")
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            raise ValueError("Global/nonlocal declarations are not allowed")
        else:
            raise ValueError(f"Forbidden operation: {type(node).__name__}")
    
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in forbidden_builtins:
            raise ValueError(f"Forbidden built-in function: {node.func.id}")

def _parse_and_validate_script(submitted_script):
    """
    Parses and validates a Python script for safe execution.
    
    Args:
        submitted_script: The script string to parse and validate
        
    Returns:
        The parsed AST tree
        
    Raises:
        ValueError: If the script contains syntax errors or forbidden operations
    """
    try:
        tree = ast.parse(submitted_script)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in the provided script: {e}")
    
    allowed_nodes = _get_allowed_nodes()
    forbidden_builtins = _get_forbidden_builtins()
    
    for node in ast.walk(tree):
        _validate_ast_node(node, allowed_nodes, forbidden_builtins)
    
    return tree

def _execute_validated_script(tree, restricted_globals):
    """
    Executes a validated AST tree in a restricted environment.
    
    Args:
        tree: The validated AST tree to execute
        restricted_globals: The restricted global environment
        
    Returns:
        The captured output as a string, or None if no output
        
    Raises:
        ValueError: If an error occurs during execution
    """
    output_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output_buffer
    
    try:
        exec(compile(tree, '<user_script>', 'exec'), restricted_globals)
        output = output_buffer.getvalue()
        return output if output else None
    except Exception as e:
        raise ValueError(f"Error during script execution: {e}")
    finally:
        sys.stdout = old_stdout

def execute_user_script(submitted_script):
    """
    Safely executes a Python script provided by the user while restricting it to only safe operations.
    
    Args:
        submitted_script: string, the script that the user submits for execution
        
    Returns:
        The output of the given script, if any, otherwise None
        
    Raises:
        ValueError: if the user script involves forbidden actions or contains syntax errors
    """
    tree = _parse_and_validate_script(submitted_script)
    restricted_globals = _get_restricted_globals()
    return _execute_validated_script(tree, restricted_globals)
