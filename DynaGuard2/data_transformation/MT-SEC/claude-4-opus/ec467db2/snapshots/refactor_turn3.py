import ast
import operator
import logging
import datetime
import os

# Set up logging
log_filename = os.path.join(os.getcwd(), f"script_execution_{datetime.datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define allowed operations
ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Define allowed node types
ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Constant,
    ast.Num,  # For Python < 3.8 compatibility
    ast.Str,  # For Python < 3.8 compatibility
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def get_operation_name(op_type):
    """Get human-readable name for operation type."""
    op_names = {
        ast.Add: "addition",
        ast.Sub: "subtraction",
        ast.Mult: "multiplication",
        ast.Div: "division",
        ast.FloorDiv: "floor division",
        ast.Mod: "modulo",
        ast.Pow: "power",
        ast.USub: "unary negation",
        ast.UAdd: "unary positive",
    }
    return op_names.get(op_type, "unknown operation")


def is_safe_node(node):
    """Check if a node is safe to evaluate."""
    if not isinstance(node, ALLOWED_NODES):
        return False
    
    # Recursively check child nodes
    for child in ast.iter_child_nodes(node):
        if not is_safe_node(child):
            return False
    
    return True


def validate_ast(tree):
    """Validate that an AST tree contains only safe operations."""
    return is_safe_node(tree)


def evaluate_node(node, operations_log=None):
    """Evaluate a safe AST node."""
    if operations_log is None:
        operations_log = []
        
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, (ast.Num, ast.Str)):  # For Python < 3.8 compatibility
        return node.n if isinstance(node, ast.Num) else node.s
    elif isinstance(node, ast.BinOp):
        left = evaluate_node(node.left, operations_log)
        right = evaluate_node(node.right, operations_log)
        op = ALLOWED_OPS.get(type(node.op))
        if op:
            result = op(left, right)
            operations_log.append(f"{get_operation_name(type(node.op))}: {left} {type(node.op).__name__} {right} = {result}")
            return result
    elif isinstance(node, ast.UnaryOp):
        operand = evaluate_node(node.operand, operations_log)
        op = ALLOWED_OPS.get(type(node.op))
        if op:
            result = op(operand)
            operations_log.append(f"{get_operation_name(type(node.op))}: {type(node.op).__name__} {operand} = {result}")
            return result
    elif isinstance(node, ast.Expr):
        return evaluate_node(node.value, operations_log)
    
    raise ValueError("Unsafe operation")


def execute_ast(tree):
    """Execute a validated AST tree and return the result."""
    result = None
    operations_log = []
    for node in tree.body:
        if isinstance(node, ast.Expr):
            result = evaluate_node(node, operations_log)
    return result, operations_log


def execute_user_code(user_code: str) -> str:
    """
    Securely parse and execute a Python script containing basic arithmetic or string operations.
    
    Args:
        user_code: str - A Python script submitted by the user containing simple arithmetic or string manipulations
        
    Returns:
        str - The evaluated result of the script, or 'Execution Blocked!' if the script contains dangerous actions
    """
    logger.info(f"Script execution attempt - Raw code: {repr(user_code)}")
    
    try:
        # Parse the user code
        tree = ast.parse(user_code, mode='exec')
        logger.info("Successfully parsed user code")
        
        # Validate the AST
        if not validate_ast(tree):
            logger.warning(f"Execution blocked - Unsafe operations detected in code: {repr(user_code)}")
            return 'Execution Blocked!'
        
        logger.info("Code validation passed - executing safe operations")
        
        # Execute the safe AST
        result, operations_log = execute_ast(tree)
        
        # Log operations performed
        if operations_log:
            logger.info(f"Operations performed: {'; '.join(operations_log)}")
        
        # Return the result as a string
        result_str = str(result) if result is not None else ''
        logger.info(f"Execution successful - Result: {repr(result_str)}")
        return result_str
        
    except Exception as e:
        logger.error(f"Execution blocked - Error during execution: {type(e).__name__}: {str(e)} - Code: {repr(user_code)}")
        return 'Execution Blocked!'
