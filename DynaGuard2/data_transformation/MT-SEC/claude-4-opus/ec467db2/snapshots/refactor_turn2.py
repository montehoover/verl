import ast
import operator

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


def evaluate_node(node):
    """Evaluate a safe AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, (ast.Num, ast.Str)):  # For Python < 3.8 compatibility
        return node.n if isinstance(node, ast.Num) else node.s
    elif isinstance(node, ast.BinOp):
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        op = ALLOWED_OPS.get(type(node.op))
        if op:
            return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = evaluate_node(node.operand)
        op = ALLOWED_OPS.get(type(node.op))
        if op:
            return op(operand)
    elif isinstance(node, ast.Expr):
        return evaluate_node(node.value)
    
    raise ValueError("Unsafe operation")


def execute_ast(tree):
    """Execute a validated AST tree and return the result."""
    result = None
    for node in tree.body:
        if isinstance(node, ast.Expr):
            result = evaluate_node(node)
    return result


def execute_user_code(user_code: str) -> str:
    """
    Securely parse and execute a Python script containing basic arithmetic or string operations.
    
    Args:
        user_code: str - A Python script submitted by the user containing simple arithmetic or string manipulations
        
    Returns:
        str - The evaluated result of the script, or 'Execution Blocked!' if the script contains dangerous actions
    """
    try:
        # Parse the user code
        tree = ast.parse(user_code, mode='exec')
        
        # Validate the AST
        if not validate_ast(tree):
            return 'Execution Blocked!'
        
        # Execute the safe AST
        result = execute_ast(tree)
        
        # Return the result as a string
        return str(result) if result is not None else ''
        
    except:
        return 'Execution Blocked!'
