import ast
import operator

# Define allowed operators
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


def validate_ast_nodes(tree):
    """
    Validate that all nodes in the AST are allowed.
    
    Args:
        tree: The AST tree to validate
        
    Returns:
        bool: True if all nodes are allowed, False otherwise
    """
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return False
    return True


def evaluate_constant_node(node):
    """
    Evaluate a constant AST node.
    
    Args:
        node: The AST constant node
        
    Returns:
        The value of the constant
        
    Raises:
        ValueError: If the constant type is not allowed
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str)):
            return node.value
        else:
            raise ValueError("Unsafe constant type")
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Str):
        return node.s
    else:
        raise ValueError("Not a constant node")


def evaluate_binary_operation(left, right, op_type):
    """
    Evaluate a binary operation.
    
    Args:
        left: Left operand
        right: Right operand
        op_type: The type of the operator
        
    Returns:
        The result of the operation
        
    Raises:
        ValueError: If the operation is invalid or unsafe
    """
    if op_type not in ALLOWED_OPERATORS:
        raise ValueError("Unsafe operator")
    
    op_func = ALLOWED_OPERATORS[op_type]
    
    # Handle string concatenation
    if isinstance(left, str) and isinstance(right, str) and op_type == ast.Add:
        return left + right
    # Handle string repetition
    elif isinstance(left, str) and isinstance(right, int) and op_type == ast.Mult:
        return left * right
    elif isinstance(left, int) and isinstance(right, str) and op_type == ast.Mult:
        return left * right
    # Handle numeric operations
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            return op_func(left, right)
        except ZeroDivisionError:
            raise ValueError("Division by zero")
    else:
        raise ValueError("Invalid operation")


def evaluate_unary_operation(operand, op_type):
    """
    Evaluate a unary operation.
    
    Args:
        operand: The operand
        op_type: The type of the operator
        
    Returns:
        The result of the operation
        
    Raises:
        ValueError: If the operation is invalid or unsafe
    """
    if op_type not in ALLOWED_OPERATORS:
        raise ValueError("Unsafe operator")
    
    op_func = ALLOWED_OPERATORS[op_type]
    
    # Only allow unary operations on numbers
    if isinstance(operand, (int, float)):
        return op_func(operand)
    else:
        raise ValueError("Invalid unary operation")


def evaluate_ast_node(node):
    """
    Recursively evaluate an AST node.
    
    Args:
        node: The AST node to evaluate
        
    Returns:
        The result of evaluating the node
        
    Raises:
        ValueError: If the node type is unsafe or evaluation fails
    """
    if isinstance(node, (ast.Constant, ast.Num, ast.Str)):
        return evaluate_constant_node(node)
        
    elif isinstance(node, ast.BinOp):
        left = evaluate_ast_node(node.left)
        right = evaluate_ast_node(node.right)
        return evaluate_binary_operation(left, right, type(node.op))
        
    elif isinstance(node, ast.UnaryOp):
        operand = evaluate_ast_node(node.operand)
        return evaluate_unary_operation(operand, type(node.op))
        
    else:
        raise ValueError("Unsafe node type")


def parse_and_validate_script(script_str):
    """
    Parse and validate a script string.
    
    Args:
        script_str: The script string to parse
        
    Returns:
        The parsed AST tree
        
    Raises:
        Exception: If parsing fails or validation fails
    """
    tree = ast.parse(script_str, mode='eval')
    
    if not validate_ast_nodes(tree):
        raise ValueError("Script contains unsafe nodes")
    
    return tree


def execute_validated_script(tree):
    """
    Execute a validated AST tree.
    
    Args:
        tree: The validated AST tree
        
    Returns:
        The result of execution as a string
        
    Raises:
        Exception: If execution fails
    """
    result = evaluate_ast_node(tree.body)
    return str(result)


def parse_user_script(script_str: str) -> str:
    """
    Safely evaluate a user-provided Python script containing only simple arithmetic and string operations.
    
    Args:
        script_str: A string containing a tiny Python script with only basic arithmetic or string operations
        
    Returns:
        Either the result of the script execution or a security warning if the input is unsafe
    """
    try:
        # Parse and validate the script
        tree = parse_and_validate_script(script_str)
        
        # Execute the validated script
        return execute_validated_script(tree)
        
    except:
        return "Execution Blocked!"
