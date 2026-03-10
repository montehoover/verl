import ast
import operator
import logging
import time
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_allowed_operations():
    """Return a dictionary of allowed AST operations mapped to their operator functions."""
    return {
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

def get_allowed_node_types():
    """Return a tuple of allowed AST node types."""
    return (
        ast.Module,
        ast.Expr,
        ast.Num,
        ast.Str,
        ast.Constant,
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

def validate_ast_safety(node, allowed_nodes):
    """
    Recursively check if all nodes in the AST are safe.
    
    Args:
        node: AST node to validate
        allowed_nodes: Tuple of allowed node types
        
    Returns:
        bool: True if safe, False otherwise
    """
    if not isinstance(node, allowed_nodes):
        return False
    
    for child in ast.iter_child_nodes(node):
        if not validate_ast_safety(child, allowed_nodes):
            return False
    
    return True

def evaluate_constant_node(node):
    """
    Evaluate a constant AST node.
    
    Args:
        node: AST constant node
        
    Returns:
        The value of the constant
        
    Raises:
        ValueError: If unsupported constant type
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str)):
            return node.value
        else:
            raise ValueError("Unsupported constant type")
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Str):
        return node.s
    else:
        raise ValueError("Not a constant node")

def evaluate_binary_operation(left, right, op, allowed_operations):
    """
    Evaluate a binary operation.
    
    Args:
        left: Left operand
        right: Right operand
        op: AST operation node
        allowed_operations: Dictionary of allowed operations
        
    Returns:
        Result of the operation
        
    Raises:
        ValueError: If invalid operation
    """
    # String concatenation
    if isinstance(op, ast.Add) and isinstance(left, str) and isinstance(right, str):
        return left + right
    # String repetition
    elif isinstance(op, ast.Mult):
        if isinstance(left, str) and isinstance(right, int):
            return left * right
        elif isinstance(left, int) and isinstance(right, str):
            return left * right
    
    # Arithmetic operations
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        op_func = allowed_operations.get(type(op))
        if op_func:
            try:
                return op_func(left, right)
            except ZeroDivisionError:
                raise ZeroDivisionError("Division by zero")
    
    raise ValueError("Invalid operation")

def evaluate_unary_operation(operand, op, allowed_operations):
    """
    Evaluate a unary operation.
    
    Args:
        operand: The operand
        op: AST operation node
        allowed_operations: Dictionary of allowed operations
        
    Returns:
        Result of the operation
        
    Raises:
        ValueError: If invalid operation
    """
    if isinstance(operand, (int, float)):
        op_func = allowed_operations.get(type(op))
        if op_func:
            return op_func(operand)
    
    raise ValueError("Invalid unary operation")

def evaluate_ast_node(node, allowed_operations):
    """
    Recursively evaluate an AST node.
    
    Args:
        node: AST node to evaluate
        allowed_operations: Dictionary of allowed operations
        
    Returns:
        Result of evaluation
        
    Raises:
        ValueError: If unsupported node type or operation
        ZeroDivisionError: If division by zero occurs
    """
    if isinstance(node, (ast.Constant, ast.Num, ast.Str)):
        return evaluate_constant_node(node)
    
    elif isinstance(node, ast.BinOp):
        left = evaluate_ast_node(node.left, allowed_operations)
        right = evaluate_ast_node(node.right, allowed_operations)
        return evaluate_binary_operation(left, right, node.op, allowed_operations)
    
    elif isinstance(node, ast.UnaryOp):
        operand = evaluate_ast_node(node.operand, allowed_operations)
        return evaluate_unary_operation(operand, node.op, allowed_operations)
    
    elif isinstance(node, ast.Expr):
        return evaluate_ast_node(node.value, allowed_operations)
    
    else:
        raise ValueError("Unsupported node type")

def parse_and_validate_code(python_code, allowed_nodes):
    """
    Parse Python code and validate its safety.
    
    Args:
        python_code: String of Python code
        allowed_nodes: Tuple of allowed node types
        
    Returns:
        Tuple of (tree, is_safe) where tree is the AST and is_safe is a boolean
    """
    try:
        # Try parsing as expression first
        tree = ast.parse(python_code, mode='eval')
        is_safe = validate_ast_safety(tree, allowed_nodes)
        return tree, is_safe, 'eval'
    except:
        try:
            # Try parsing as module
            tree = ast.parse(python_code, mode='exec')
            is_safe = validate_ast_safety(tree, allowed_nodes)
            # For exec mode, only support single expression statements
            if is_safe and len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                return tree, is_safe, 'exec'
            else:
                return tree, False, 'exec'
        except:
            return None, False, None

def execute_safe_code(tree, mode, allowed_operations):
    """
    Execute validated safe AST.
    
    Args:
        tree: Validated AST tree
        mode: 'eval' or 'exec' mode
        allowed_operations: Dictionary of allowed operations
        
    Returns:
        Result of execution as string
        
    Raises:
        Exception: Any exception during execution
    """
    if mode == 'eval':
        result = evaluate_ast_node(tree.body, allowed_operations)
    else:  # mode == 'exec'
        result = evaluate_ast_node(tree.body[0], allowed_operations)
    
    return str(result)

def get_script_hash(script):
    """Generate a short hash of the script for logging purposes."""
    return hashlib.md5(script.encode()).hexdigest()[:8]

def get_script_preview(script, max_length=50):
    """Get a preview of the script for logging."""
    script = script.strip()
    if len(script) <= max_length:
        return script.replace('\n', ' ')
    return script[:max_length].replace('\n', ' ') + '...'

def run_user_code(python_code: str) -> str:
    """
    Securely evaluates a user-supplied Python script limited to basic arithmetic and string operations.
    
    Args:
        python_code: str - A user-supplied Python script limited to simple arithmetic and string operations
        
    Returns:
        str - The result of evaluating the script or 'Execution Blocked!' if unsafe
    """
    start_time = time.time()
    script_hash = get_script_hash(python_code)
    script_preview = get_script_preview(python_code)
    
    logger.info(f"Script evaluation started - Hash: {script_hash}, Preview: '{script_preview}'")
    
    allowed_operations = get_allowed_operations()
    allowed_nodes = get_allowed_node_types()
    
    tree, is_safe, mode = parse_and_validate_code(python_code, allowed_nodes)
    
    if not is_safe or tree is None:
        execution_time = time.time() - start_time
        logger.warning(f"Script blocked - Hash: {script_hash}, Reason: Failed safety validation, "
                      f"Execution time: {execution_time:.4f}s, Script: '{script_preview}'")
        return "Execution Blocked!"
    
    try:
        result = execute_safe_code(tree, mode, allowed_operations)
        execution_time = time.time() - start_time
        logger.info(f"Script executed successfully - Hash: {script_hash}, Result: '{result}', "
                   f"Execution time: {execution_time:.4f}s, Mode: {mode}")
        return result
    except ZeroDivisionError:
        execution_time = time.time() - start_time
        logger.warning(f"Script blocked - Hash: {script_hash}, Reason: Division by zero, "
                      f"Execution time: {execution_time:.4f}s, Script: '{script_preview}'")
        return "Execution Blocked!"
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Script execution failed - Hash: {script_hash}, Error: {type(e).__name__}: {str(e)}, "
                    f"Execution time: {execution_time:.4f}s, Script: '{script_preview}'")
        return "Execution Blocked!"
