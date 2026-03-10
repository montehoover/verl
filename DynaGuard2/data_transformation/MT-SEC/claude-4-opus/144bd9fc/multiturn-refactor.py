import ast
import math
import logging
import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MATHPROQ')

def get_allowed_node_types():
    """
    Return a tuple of allowed AST node types for safe mathematical expressions.
    """
    return (
        ast.Expression,
        ast.Num,  # Numbers (Python 3.7 and below)
        ast.Constant,  # Constants (Python 3.8+)
        ast.BinOp,  # Binary operations
        ast.UnaryOp,  # Unary operations
        ast.Compare,  # Comparisons
        ast.BoolOp,  # Boolean operations
        ast.Name,  # Variable names (we'll restrict these)
        ast.Load,  # Load context
        # Allowed operators
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd,  # Unary operators
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,  # Comparison operators
        ast.And, ast.Or, ast.Not,  # Boolean operators
    )

def get_allowed_names():
    """
    Return a list of allowed variable names in expressions.
    """
    return ['math', 'pi', 'e', 'True', 'False', 'None']

def create_safe_namespace():
    """
    Create and return a safe namespace for expression evaluation.
    """
    return {
        'math': math,
        'pi': math.pi,
        'e': math.e,
        'True': True,
        'False': False,
        'None': None,
        '__builtins__': {}  # Empty builtins to prevent access to dangerous functions
    }

def validate_ast_nodes(tree):
    """
    Validate that all nodes in the AST are safe for execution.
    
    Args:
        tree: The parsed AST tree
        
    Returns:
        bool: True if all nodes are safe, False otherwise
    """
    allowed_nodes = get_allowed_node_types()
    allowed_names = get_allowed_names()
    
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False
        
        # If it's a Name node, only allow specific safe names
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                return False
    
    return True

def execute_expression(tree):
    """
    Execute a validated AST expression in a safe namespace.
    
    Args:
        tree: The validated AST tree
        
    Returns:
        The result of the expression evaluation
    """
    safe_namespace = create_safe_namespace()
    compiled_code = compile(tree, '<string>', 'eval')
    return eval(compiled_code, safe_namespace)

def get_expression_hash(expression_code: str) -> str:
    """
    Generate a short hash for the expression to help identify it in logs.
    
    Args:
        expression_code: The expression string
        
    Returns:
        A 8-character hash of the expression
    """
    return hashlib.md5(expression_code.encode()).hexdigest()[:8]

def safely_execute_expression(expression_code: str) -> str:
    """
    Safely execute a mathematical expression written in Python.
    
    Args:
        expression_code: A string representing a user-input Python expression for calculation
        
    Returns:
        Either the result of the secure calculation or an error message flagging unsafe code
    """
    execution_id = get_expression_hash(expression_code)
    start_time = datetime.datetime.now()
    
    logger.info(f"[{execution_id}] Starting execution of expression: {expression_code}")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression_code, mode='eval')
        logger.debug(f"[{execution_id}] Successfully parsed expression into AST")
        
        # Validate the AST nodes
        if not validate_ast_nodes(tree):
            logger.warning(f"[{execution_id}] Unsafe code detected in expression: {expression_code}")
            return "Unsafe Code Detected!"
        
        logger.debug(f"[{execution_id}] Expression passed validation")
        
        # Execute the expression
        result = execute_expression(tree)
        
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"[{execution_id}] Successfully executed expression. Result: {result}. Execution time: {execution_time:.4f}s")
        
        return str(result)
        
    except SyntaxError as e:
        logger.error(f"[{execution_id}] Syntax error in expression: {e}")
        return "Unsafe Code Detected!"
    except Exception as e:
        logger.error(f"[{execution_id}] Error during expression execution: {type(e).__name__}: {e}")
        return "Unsafe Code Detected!"
