import ast
import logging

# Configure logging
logger = logging.getLogger(__name__)

# List of allowed node types for safe execution
ALLOWED_NODES = {
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
ALLOWED_BUILTINS = {
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
    'chr', 'complex', 'dict', 'divmod', 'enumerate', 'filter', 'float',
    'format', 'frozenset', 'hash', 'hex', 'int', 'isinstance', 'issubclass',
    'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'oct', 'ord',
    'pow', 'print', 'range', 'repr', 'reversed', 'round', 'set',
    'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
}

# List of explicitly disallowed built-in functions
DISALLOWED_BUILTINS = {
    'eval', 'exec', 'compile', '__import__', 'open', 
    'input', 'breakpoint', 'help', 'exit', 'quit'
}


def parse_script(user_script: str) -> ast.AST:
    """
    Parse the user script into an AST.
    
    Args:
        user_script: str, the Python script provided by the user
        
    Returns:
        ast.AST: The parsed abstract syntax tree
        
    Raises:
        ValueError: if the script contains syntax errors
    """
    try:
        tree = ast.parse(user_script, mode='exec')
        logger.debug(f"Successfully parsed script: {user_script[:100]}{'...' if len(user_script) > 100 else ''}")
        return tree
    except SyntaxError as e:
        logger.error(f"Syntax error in script: {e}")
        raise ValueError(f"Syntax error in script: {e}")


def validate_node(node: ast.AST) -> None:
    """
    Validate a single AST node for security.
    
    Args:
        node: ast.AST, the node to validate
        
    Raises:
        ValueError: if the node represents a disallowed operation
    """
    # Check if node type is allowed
    if type(node) not in ALLOWED_NODES:
        # Special handling for Import and ImportFrom
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            logger.warning(f"Blocked import statement in script")
            raise ValueError("Import statements are not allowed")
        # Special handling for FunctionDef and ClassDef
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            logger.warning(f"Blocked function/class definition in script")
            raise ValueError("Function and class definitions are not allowed")
        # Special handling for Global and Nonlocal
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            logger.warning(f"Blocked global/nonlocal statement in script")
            raise ValueError("Global and nonlocal statements are not allowed")
        else:
            logger.warning(f"Blocked disallowed operation: {type(node).__name__}")
            raise ValueError(f"Disallowed operation: {type(node).__name__}")
    
    # Check for disallowed built-in functions
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in DISALLOWED_BUILTINS:
            logger.warning(f"Blocked disallowed built-in function: {node.func.id}")
            raise ValueError(f"Disallowed built-in function: {node.func.id}")
        if node.func.id not in ALLOWED_BUILTINS and node.func.id in __builtins__:
            logger.warning(f"Blocked non-whitelisted built-in function: {node.func.id}")
            raise ValueError(f"Disallowed built-in function: {node.func.id}")
    
    # Check for attribute access on modules or dangerous objects
    if isinstance(node, ast.Attribute):
        if node.attr.startswith('_'):
            logger.warning(f"Blocked access to private attribute: {node.attr}")
            raise ValueError("Access to private attributes is not allowed")


def validate_script(tree: ast.AST) -> None:
    """
    Validate the entire AST for security.
    
    Args:
        tree: ast.AST, the parsed abstract syntax tree
        
    Raises:
        ValueError: if the script contains disallowed operations
    """
    for node in ast.walk(tree):
        validate_node(node)
    logger.debug("Script validation completed successfully")


def create_restricted_environment() -> dict:
    """
    Create a restricted execution environment.
    
    Returns:
        dict: A dictionary with restricted built-in functions
    """
    restricted_env = {
        '__builtins__': {
            name: __builtins__[name] 
            for name in ALLOWED_BUILTINS 
            if name in __builtins__
        }
    }
    logger.debug(f"Created restricted environment with {len(restricted_env['__builtins__'])} allowed builtins")
    return restricted_env


def execute_script(tree: ast.AST, user_script: str, restricted_globals: dict):
    """
    Execute the validated script and return any result.
    
    Args:
        tree: ast.AST, the validated abstract syntax tree
        user_script: str, the original script text
        restricted_globals: dict, the restricted execution environment
        
    Returns:
        The result of the script if any, or None
        
    Raises:
        ValueError: if an error occurs during execution
    """
    try:
        exec(tree, restricted_globals)
        
        # Check if there's a result to return
        if user_script.strip().split('\n')[-1].strip() and not user_script.strip().endswith(':'):
            # Try to evaluate the last line as an expression
            try:
                last_line = user_script.strip().split('\n')[-1].strip()
                result = eval(last_line, restricted_globals)
                logger.info(f"Script executed successfully with result: {result}")
                return result
            except:
                logger.info("Script executed successfully with no return value")
                return None
        logger.info("Script executed successfully with no return value")
        return None
    except Exception as e:
        logger.error(f"Error executing script: {e}")
        raise ValueError(f"Error executing script: {e}")


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
    logger.info(f"Starting script execution for script of length {len(user_script)} characters")
    
    try:
        # Parse the script
        tree = parse_script(user_script)
        
        # Validate the script
        validate_script(tree)
        
        # Create restricted environment
        restricted_globals = create_restricted_environment()
        
        # Execute the script and return result
        result = execute_script(tree, user_script, restricted_globals)
        logger.info("Script execution completed successfully")
        return result
        
    except ValueError as e:
        logger.error(f"Script execution failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during script execution: {e}")
        raise ValueError(f"Unexpected error: {e}")
