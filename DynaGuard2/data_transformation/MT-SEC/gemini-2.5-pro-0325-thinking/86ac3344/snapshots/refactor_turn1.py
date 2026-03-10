import ast

def secure_eval_expression(exp: str):
    """
    Evaluates a mathematical expression string securely.

    Args:
        exp: A user-provided string containing the mathematical operation.

    Returns:
        The evaluated result of the provided mathematical expression.

    Raises:
        ValueError: If invalid characters are detected, or if the operation
                    involves any unsafe or restricted behavior.
    """
    # Allowed AST node types
    allowed_nodes = {
        ast.Expression, ast.Num, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
        ast.Constant # Python 3.8+ for numbers, strings, None, True, False
    }
    # For older Python versions, ast.Num is used for numbers.
    # ast.NameConstant for True, False, None (Python < 3.8)
    # ast.Str for strings (if you decide to allow them, currently not in scope for math)

    try:
        node = ast.parse(exp, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")

    for sub_node in ast.walk(node):
        if not isinstance(sub_node, tuple(allowed_nodes)):
            raise ValueError(f"Unsafe operation or character: {type(sub_node).__name__}")

    # Ensure the expression is not empty or just whitespace
    if not exp.strip():
        raise ValueError("Expression cannot be empty.")

    # Evaluate the expression using a restricted environment
    # No builtins or globals are passed to eval, making it safer.
    # The `compile` step ensures the AST is valid executable code.
    try:
        code = compile(node, '<string>', 'eval')
        # Provide a limited scope for eval
        # For simple arithmetic, an empty dict for globals and locals is usually sufficient.
        # If you need math functions like sqrt, sin, etc., you'd add them to a safe_globals dict.
        # e.g., safe_globals = {'sqrt': math.sqrt}
        # For now, only basic arithmetic is supported by the AST check.
        return eval(code, {"__builtins__": {}}, {})
    except Exception as e:
        # Catch any other evaluation errors (e.g., division by zero)
        raise ValueError(f"Error during evaluation: {e}")
