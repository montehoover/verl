import ast
import logging

# Configure logger
logger = logging.getLogger(__name__)


def _parse_and_validate_snippet(snippet_string: str) -> ast.AST:
    """
    Parse Python code and validate it doesn't contain forbidden operations.
    
    Args:
        snippet_string: The Python code to parse and validate
        
    Returns:
        The parsed AST tree
        
    Raises:
        ValueError: If the code contains invalid syntax or forbidden operations
    """
    logger.debug(f"Parsing snippet: {snippet_string[:100]}{'...' if len(snippet_string) > 100 else ''}")
    
    # Parse the code
    try:
        tree = ast.parse(snippet_string)
        logger.debug("Successfully parsed snippet")
    except SyntaxError as e:
        logger.error(f"Failed to parse snippet: {e}")
        raise ValueError(f"Invalid Python syntax: {e}")
    
    # Define forbidden node types
    forbidden_nodes = (
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
        ast.AsyncFor,
        ast.AsyncFunctionDef,
    )
    
    # Define forbidden function names
    forbidden_names = {
        '__import__', 'eval', 'exec', 'compile', 'open', 'file', 'input',
        'raw_input', 'execfile', 'reload', 'vars', 'locals', 'globals',
        'dir', 'help', 'type', 'getattr', 'setattr', 'delattr', 'hasattr',
        'callable', 'classmethod', 'staticmethod', 'property', 'super',
        'isinstance', 'issubclass', 'print', 'exit', 'quit'
    }
    
    # Define forbidden attributes
    forbidden_attrs = {
        '__getattr__', '__setattr__', '__delattr__', '__getattribute__',
        '__class__', '__dict__', '__module__', '__bases__', '__subclasses__'
    }
    
    # Check for forbidden operations
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            logger.warning(f"Detected forbidden operation: {node.__class__.__name__}")
            raise ValueError(f"Forbidden operation: {node.__class__.__name__}")
        
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            logger.warning(f"Detected forbidden function: {node.id}")
            raise ValueError(f"Forbidden function: {node.id}")
        
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                logger.warning(f"Detected forbidden function call: {node.func.id}")
                raise ValueError(f"Forbidden function call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attrs:
                logger.warning(f"Detected forbidden attribute access: {node.func.attr}")
                raise ValueError(f"Forbidden attribute access: {node.func.attr}")
    
    logger.debug("Snippet validation passed")
    return tree


def _create_safe_environment() -> dict:
    """
    Create a safe execution environment with limited builtins.
    
    Returns:
        A dictionary containing safe global variables
    """
    logger.debug("Creating safe execution environment")
    
    safe_builtins = {
        'abs': abs,
        'all': all,
        'any': any,
        'bin': bin,
        'bool': bool,
        'chr': chr,
        'complex': complex,
        'dict': dict,
        'divmod': divmod,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'format': format,
        'frozenset': frozenset,
        'hex': hex,
        'int': int,
        'iter': iter,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'next': next,
        'oct': oct,
        'ord': ord,
        'pow': pow,
        'range': range,
        'reversed': reversed,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'zip': zip,
        'True': True,
        'False': False,
        'None': None,
    }
    
    logger.debug(f"Safe environment created with {len(safe_builtins)} allowed builtins")
    return {'__builtins__': safe_builtins}


def _execute_snippet(snippet_string: str, safe_globals: dict) -> tuple:
    """
    Execute a Python snippet in a safe environment.
    
    Args:
        snippet_string: The Python code to execute
        safe_globals: The safe global environment dictionary
        
    Returns:
        A tuple of (result, locals_dict) where result is the output value
        and locals_dict contains any variables created during execution
        
    Raises:
        ValueError: If execution fails
    """
    safe_locals = {}
    
    logger.debug("Starting snippet execution")
    
    try:
        # Compile and execute the code
        compiled_code = compile(snippet_string, '<string>', 'exec')
        exec(compiled_code, safe_globals, safe_locals)
        logger.debug(f"Snippet executed successfully. Local variables created: {list(safe_locals.keys())}")
        
        # Try to evaluate as a single expression to get return value
        try:
            expr_tree = ast.parse(snippet_string.strip(), mode='eval')
            result = eval(compile(expr_tree, '<string>', 'eval'), safe_globals, safe_locals)
            logger.debug(f"Snippet evaluated as expression with result: {result}")
            return result, safe_locals
        except:
            logger.debug("Snippet is not a single expression")
            return None, safe_locals
            
    except Exception as e:
        logger.error(f"Error during snippet execution: {e}")
        raise ValueError(f"Error executing snippet: {e}")


def run_python_snippet(snippet_string: str):
    """
    Executes a Python code snippet safely, ensuring only benign actions are carried out.
    
    Args:
        snippet_string: str - The user-provided Python code to be executed
        
    Returns:
        The resulting value if the code produces any output, or None otherwise
        
    Raises:
        ValueError: If the snippet attempts forbidden operations or contains invalid code
    """
    logger.info(f"Starting execution of snippet: {snippet_string[:50]}{'...' if len(snippet_string) > 50 else ''}")
    
    # Parse and validate the snippet
    _parse_and_validate_snippet(snippet_string)
    
    # Create safe execution environment
    safe_globals = _create_safe_environment()
    
    # Execute the snippet
    result, safe_locals = _execute_snippet(snippet_string, safe_globals)
    
    # Return the result or last assigned value
    if result is not None:
        logger.info(f"Snippet execution completed with result: {result}")
        return result
    elif safe_locals:
        final_value = list(safe_locals.values())[-1]
        logger.info(f"Snippet execution completed with last assigned value: {final_value}")
        return final_value
    else:
        logger.info("Snippet execution completed with no return value")
        return None
