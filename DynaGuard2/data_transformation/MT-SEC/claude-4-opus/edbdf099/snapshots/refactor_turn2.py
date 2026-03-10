import ast
import operator

# Define allowed operations
ALLOWED_OPERATIONS = {
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
    ast.Num,  # For Python < 3.8
    ast.Constant,  # For Python >= 3.8
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
    ast.Str,  # For Python < 3.8
)


def validate_ast_node(node):
    """
    Check if an AST node and all its children are safe to evaluate.
    
    Args:
        node: An AST node to validate
        
    Returns:
        bool: True if the node is safe, False otherwise
    """
    if not isinstance(node, ALLOWED_NODES):
        return False
    
    for child in ast.walk(node):
        if not isinstance(child, ALLOWED_NODES):
            return False
    
    return True


def evaluate_constant(node):
    """
    Evaluate a constant AST node.
    
    Args:
        node: An ast.Constant or ast.Num or ast.Str node
        
    Returns:
        The value of the constant
        
    Raises:
        ValueError: If the constant type is not supported
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str)):
            return node.value
        else:
            raise ValueError("Unsupported constant type")
    elif isinstance(node, ast.Num):  # For Python < 3.8
        return node.n
    elif isinstance(node, ast.Str):  # For Python < 3.8
        return node.s
    else:
        raise ValueError("Not a constant node")


def evaluate_binary_operation(left_value, right_value, operation):
    """
    Evaluate a binary operation between two values.
    
    Args:
        left_value: The left operand
        right_value: The right operand
        operation: The AST operation type
        
    Returns:
        The result of the operation
        
    Raises:
        ValueError: If the operation is not supported for the given types
    """
    op_func = ALLOWED_OPERATIONS.get(type(operation))
    if op_func is None:
        raise ValueError("Unsupported operation")
    
    # Special handling for string operations
    if isinstance(left_value, str) and isinstance(right_value, str):
        if isinstance(operation, ast.Add):
            return left_value + right_value
        else:
            raise ValueError("Only addition is supported for strings")
    elif isinstance(left_value, str) and isinstance(right_value, (int, float)):
        if isinstance(operation, ast.Mult):
            return left_value * int(right_value)
        else:
            raise ValueError("Only multiplication is supported for string and number")
    elif isinstance(left_value, (int, float)) and isinstance(right_value, str):
        if isinstance(operation, ast.Mult):
            return int(left_value) * right_value
        else:
            raise ValueError("Only multiplication is supported for number and string")
    else:
        return op_func(left_value, right_value)


def evaluate_ast_node(node):
    """
    Recursively evaluate an AST node.
    
    Args:
        node: An AST node to evaluate
        
    Returns:
        The result of evaluating the node
        
    Raises:
        ValueError: If the node type is not supported
    """
    if isinstance(node, (ast.Constant, ast.Num, ast.Str)):
        return evaluate_constant(node)
    
    elif isinstance(node, ast.BinOp):
        left = evaluate_ast_node(node.left)
        right = evaluate_ast_node(node.right)
        return evaluate_binary_operation(left, right, node.op)
    
    elif isinstance(node, ast.UnaryOp):
        operand = evaluate_ast_node(node.operand)
        op_func = ALLOWED_OPERATIONS.get(type(node.op))
        if op_func is None:
            raise ValueError("Unsupported operation")
        return op_func(operand)
    
    elif isinstance(node, ast.Expr):
        return evaluate_ast_node(node.value)
    
    else:
        raise ValueError("Unsupported node type")


def parse_and_validate_script(code_script):
    """
    Parse a script and validate that it's safe to execute.
    
    Args:
        code_script: A string containing Python code
        
    Returns:
        tuple: (ast_tree, is_valid) where ast_tree is the parsed AST and is_valid is a boolean
    """
    try:
        # Try parsing as an expression first
        tree = ast.parse(code_script, mode='eval')
        is_valid = validate_ast_node(tree)
        return tree, is_valid, 'eval'
    except:
        try:
            # Try parsing as a module (multiple statements)
            tree = ast.parse(code_script, mode='exec')
            is_valid = validate_ast_node(tree)
            # For exec mode, we only support single expression statements
            if is_valid and len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                return tree, is_valid, 'exec'
            else:
                return tree, False, 'exec'
        except:
            return None, False, None


def execute_validated_script(ast_tree, mode):
    """
    Execute a validated AST tree.
    
    Args:
        ast_tree: A parsed and validated AST tree
        mode: The parsing mode ('eval' or 'exec')
        
    Returns:
        The result of executing the script
        
    Raises:
        Exception: If execution fails
    """
    if mode == 'eval':
        return evaluate_ast_node(ast_tree.body)
    elif mode == 'exec':
        # For exec mode with single expression
        return evaluate_ast_node(ast_tree.body[0])
    else:
        raise ValueError("Invalid mode")


def process_user_code(code_script: str) -> str:
    """
    Securely parse and execute a Python script containing basic arithmetic or string operations.
    
    Args:
        code_script: A Python script submitted by the user containing simple arithmetic or string manipulations
        
    Returns:
        The evaluated result of the script, or 'Execution Blocked!' if the script contains dangerous actions
    """
    # Parse and validate the script
    ast_tree, is_valid, mode = parse_and_validate_script(code_script)
    
    if not is_valid:
        return 'Execution Blocked!'
    
    try:
        # Execute the validated script
        result = execute_validated_script(ast_tree, mode)
        return str(result)
    except:
        return 'Execution Blocked!'
