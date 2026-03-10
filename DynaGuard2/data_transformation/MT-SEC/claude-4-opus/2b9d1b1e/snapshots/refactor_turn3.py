import ast
import operator
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='script_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Define allowed operations
ALLOWED_OPERATORS = {
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

ALLOWED_FUNCTIONS = {
    'len': len,
    'str': str,
    'int': int,
    'float': float,
}


def validate_ast_node(node):
    """
    Validate if an AST node is safe to evaluate.
    
    Args:
        node: An AST node to validate
        
    Returns:
        bool: True if the node is safe, False otherwise
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str))
    elif isinstance(node, ast.BinOp):
        return (type(node.op) in ALLOWED_OPERATORS and 
                validate_ast_node(node.left) and 
                validate_ast_node(node.right))
    elif isinstance(node, ast.UnaryOp):
        return (type(node.op) in ALLOWED_OPERATORS and 
                validate_ast_node(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS:
            return all(validate_ast_node(arg) for arg in node.args)
        return False
    elif isinstance(node, ast.Name):
        return False  # No variable access allowed
    else:
        return False


def execute_ast_node(node):
    """
    Execute a validated safe AST node.
    
    Args:
        node: A validated AST node to execute
        
    Returns:
        The result of evaluating the node
    """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = execute_ast_node(node.left)
        right = execute_ast_node(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = execute_ast_node(node.operand)
        return ALLOWED_OPERATORS[type(node.op)](operand)
    elif isinstance(node, ast.Call):
        func = ALLOWED_FUNCTIONS[node.func.id]
        args = [execute_ast_node(arg) for arg in node.args]
        return func(*args)


def parse_code_string(code_str):
    """
    Parse a code string into an AST.
    
    Args:
        code_str: String containing Python code
        
    Returns:
        AST node or None if parsing fails
    """
    try:
        return ast.parse(code_str, mode='eval')
    except:
        return None


def log_evaluation(code_str, status, result=None):
    """
    Log the evaluation attempt with details.
    
    Args:
        code_str: The input script
        status: 'SAFE' or 'BLOCKED' 
        result: The evaluation result (if safe)
    """
    log_entry = f"Script: {repr(code_str)} | Status: {status}"
    if result is not None:
        log_entry += f" | Result: {result}"
    logging.info(log_entry)


def evaluate_user_code(code_str: str) -> str:
    """
    Safely evaluate a user-supplied Python script limited to basic arithmetic and string operations.
    
    Args:
        code_str: A user-supplied Python script limited to simple arithmetic and string operations
        
    Returns:
        The result of evaluating the script or 'Execution Blocked!' if unsafe
    """
    # Parse the code into an AST
    tree = parse_code_string(code_str)
    if tree is None:
        log_evaluation(code_str, 'BLOCKED')
        return 'Execution Blocked!'
    
    # Validate the AST
    if not validate_ast_node(tree.body):
        log_evaluation(code_str, 'BLOCKED')
        return 'Execution Blocked!'
    
    # Execute the validated AST
    try:
        result = execute_ast_node(tree.body)
        result_str = str(result)
        log_evaluation(code_str, 'SAFE', result_str)
        return result_str
    except:
        log_evaluation(code_str, 'BLOCKED')
        return 'Execution Blocked!'
