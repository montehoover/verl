import ast

def run_user_code(python_code: str) -> str:
    """
    Securely evaluates a user-supplied Python script string, limited to basic
    arithmetic and string manipulations.

    Args:
        python_code: A string containing the Python code to evaluate.
                     Expected to be a single expression.

    Returns:
        The string representation of the evaluation result if successful and safe,
        or "Execution Blocked!" if the script is unsafe or an error occurs.
    """
    ALLOWED_NODE_TYPES_BASE = (
        ast.Expression,  # The root of an expression.
        ast.Constant,    # Literals: numbers, strings, True, False, None.
    )
    # Whitelist of allowed binary operators (e.g., +, -, *, /).
    ALLOWED_BIN_OPS = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow
    )
    # Whitelist of allowed unary operators (e.g., unary -).
    ALLOWED_UNARY_OPS = (
        ast.UAdd, ast.USub
    )

    try:
        # Parse the Python code string into an AST.
        # mode='eval' ensures it's a single expression.
        tree = ast.parse(python_code, mode='eval')
    except SyntaxError:
        return "Execution Blocked!"

    # Validate all nodes in the AST.
    for node in ast.walk(tree):
        if isinstance(node, ALLOWED_NODE_TYPES_BASE):
            continue  # Basic expression/constant nodes are allowed.
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, ALLOWED_BIN_OPS):
                # Disallow binary operations not in the whitelist (e.g., bitwise ops).
                return "Execution Blocked!"
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ALLOWED_UNARY_OPS):
                # Disallow unary operations not in the whitelist (e.g., bitwise not).
                return "Execution Blocked!"
        else:
            # Any other AST node type is disallowed (e.g., Call, Name, Attribute, List, Dict, etc.).
            return "Execution Blocked!"

    try:
        # Compile the AST into a code object.
        code_object = compile(tree, filename='<user_script>', mode='eval')
        
        # Evaluate the code object in a restricted environment.
        # Empty __builtins__ and locals to prevent access to unsafe functions/variables.
        result = eval(code_object, {'__builtins__': {}}, {})
        
        return str(result)
    except Exception:
        # Catch any runtime errors during evaluation (e.g., ZeroDivisionError, TypeError)
        # or other issues.
        return "Execution Blocked!"
