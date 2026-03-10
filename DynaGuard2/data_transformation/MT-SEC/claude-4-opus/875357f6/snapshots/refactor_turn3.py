import ast
import io
import sys
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

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

def _get_forbidden_node_error_message(node):
    """Returns appropriate error message for forbidden node types."""
    forbidden_node_map = {
        (ast.Import, ast.ImportFrom): "Import statements are not allowed",
        (ast.FunctionDef, ast.AsyncFunctionDef): "Function definitions are not allowed",
        (ast.ClassDef,): "Class definitions are not allowed",
        (ast.Try,): "Try/except blocks are not allowed",
        (ast.Raise,): "Raise statements are not allowed",
        (ast.Assert,): "Assert statements are not allowed",
        (ast.Global, ast.Nonlocal): "Global/nonlocal declarations are not allowed"
    }
    
    for node_types, message in forbidden_node_map.items():
        if isinstance(node, node_types):
            return message
    
    return f"Forbidden operation: {type(node).__name__}"

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
        error_message = _get_forbidden_node_error_message(node)
        logger.warning(f"Validation failed: {error_message}")
        raise ValueError(error_message)
    
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_builtins:
        error_message = f"Forbidden built-in function: {node.func.id}"
        logger.warning(f"Validation failed: {error_message}")
        raise ValueError(error_message)

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
    logger.debug(f"Parsing script of length {len(submitted_script)}")
    
    try:
        tree = ast.parse(submitted_script)
    except SyntaxError as e:
        logger.error(f"Syntax error in script: {e}")
        raise ValueError(f"Syntax error in the provided script: {e}")
    
    allowed_nodes = _get_allowed_nodes()
    forbidden_builtins = _get_forbidden_builtins()
    
    logger.debug("Validating AST nodes")
    for node in ast.walk(tree):
        _validate_ast_node(node, allowed_nodes, forbidden_builtins)
    
    logger.debug("Script validation successful")
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
        logger.debug("Executing validated script")
        exec(compile(tree, '<user_script>', 'exec'), restricted_globals)
        output = output_buffer.getvalue()
        logger.debug(f"Script executed successfully, output length: {len(output) if output else 0}")
        return output if output else None
    except Exception as e:
        logger.error(f"Error during script execution: {e}")
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
    timestamp = datetime.now().isoformat()
    logger.info(f"Script submission at {timestamp}, length: {len(submitted_script)}")
    
    try:
        tree = _parse_and_validate_script(submitted_script)
        restricted_globals = _get_restricted_globals()
        result = _execute_validated_script(tree, restricted_globals)
        
        logger.info(f"Script executed successfully at {timestamp}, output: {'Yes' if result else 'No output'}")
        if result:
            logger.debug(f"Output preview: {result[:100]}..." if len(result) > 100 else f"Output: {result}")
        
        return result
    except ValueError as e:
        logger.error(f"Script execution failed at {timestamp}: {e}")
        raise
