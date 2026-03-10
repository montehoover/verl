import ast
import operator

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
        return 'Execution Blocked!'
    
    # Validate the AST
    if not validate_ast_node(tree.body):
        return 'Execution Blocked!'
    
    # Execute the validated AST
    try:
        result = execute_ast_node(tree.body)
        return str(result)
    except:
        return 'Execution Blocked!'
