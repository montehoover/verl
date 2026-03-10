import ast
import operator as op

# Supported operations
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos, # For unary plus, though it usually doesn't change the number
}

ALLOWED_NODES = (
    ast.Expression, 
    ast.Num, # Deprecated in Python 3.8, use ast.Constant
    ast.Constant, # For numbers and None
    ast.BinOp, 
    ast.UnaryOp,
    ast.Call, # To allow functions like float() if needed, but we'll restrict names
    ast.Name, # To allow names like 'float'
    ast.Load, # Context for loading a variable
)

# Whitelist of allowed names (e.g., built-in functions or constants if any)
# For now, we are only evaluating numbers and basic operations, so this can be empty
# or restricted to very specific safe functions if needed in the future.
ALLOWED_NAMES = {'float': float}


def _eval_node(node):
    """
    Recursively evaluate an AST node.
    """
    if isinstance(node, ast.Constant): # Handles numbers, strings, None, True, False
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            # Disallow other constants like strings for mathematical expressions
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Num): # For older Python versions (before 3.8)
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        operator_func = ALLOWED_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op)}")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero")
        return operator_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        operator_func = ALLOWED_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        return operator_func(operand)
    elif isinstance(node, ast.Name):
        if node.id in ALLOWED_NAMES:
            return ALLOWED_NAMES[node.id]
        raise ValueError(f"Unsupported name: {node.id}")
    elif isinstance(node, ast.Call):
        # This part is tricky and needs to be very careful.
        # For now, let's only allow float() if it's explicitly in ALLOWED_NAMES
        # and it's called with a single numeric argument.
        func = _eval_node(node.func)
        if func == float and len(node.args) == 1:
            arg_val = _eval_node(node.args[0])
            if isinstance(arg_val, (int, float)):
                return float(arg_val)
        raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")
    
    # Check if the node type is allowed at all
    if not isinstance(node, ALLOWED_NODES):
        raise ValueError(f"Unsupported AST node type: {type(node)}")
    
    # Fallback for nodes that are allowed but not directly evaluated (like ast.Expression, ast.Load)
    # This might need more specific handling if complex expressions are supported.
    # For simple cases, this might not be hit if the structure is always Expression(BinOp/UnaryOp/Constant).
    raise ValueError(f"Cannot evaluate node type: {type(node)}")


def perform_safe_math(expression: str) -> float:
    """
    Evaluates a mathematical expression string safely.

    Args:
        expression: The mathematical expression string (e.g., "5 + 7 * (3 - 1) / 2").

    Returns:
        The result of the evaluation as a float.

    Raises:
        ValueError: If the expression contains unsafe characters,
                    is improperly formatted, or uses unsupported operations/nodes.
        SyntaxError: If the expression is not valid Python syntax.
    """
    if not isinstance(expression, str):
        raise TypeError("Expression must be a string.")
    if not expression.strip():
        raise ValueError("Expression cannot be empty.")

    try:
        # 1. Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}")

    # 2. Validate all nodes in the AST
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Unsafe or unsupported node type in expression: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            raise ValueError(f"Unsafe or unsupported name in expression: {node.id}")
        if isinstance(node, ast.Call):
            # Further restrict calls if necessary, e.g., check node.func
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_NAMES):
                 raise ValueError(f"Unsafe or unsupported function call: {ast.dump(node.func)}")


    # 3. Evaluate the parsed and validated AST
    # The body of an 'eval' mode AST is an Expression node, whose value is the actual expression.
    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError: # Should be caught by _eval_node, but as a safeguard
        raise ValueError("Division by zero")
    except ValueError as e: # Catch specific ValueErrors from _eval_node
        raise e 
    except Exception as e: # Catch any other unexpected errors during evaluation
        raise ValueError(f"Error evaluating expression: {e}")

    if not isinstance(result, (int, float)):
        raise ValueError(f"Expression did not evaluate to a number. Got: {type(result)}")
    
    return float(result)
