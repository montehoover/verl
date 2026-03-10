import ast

def calculate_expression(expression):
    """
    Safely evaluate a simple arithmetic expression string.
    Supports: addition (+), subtraction (-), multiplication (*), division (/), and parentheses.
    Returns the numerical result, or 'Execution Blocked!' if invalid/unsafe.
    """
    if not isinstance(expression, str):
        return 'Execution Blocked!'

    try:
        node = ast.parse(expression, mode='eval')
    except Exception:
        return 'Execution Blocked!'

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)

            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            # Disallow any other binary operators (e.g., **, //, %, bitwise, etc.)
            raise ValueError("Unsupported operator")

        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            # Disallow other unary operators
            raise ValueError("Unsupported unary operator")

        # Support numeric literals only (int/float), reject bool and others
        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, bool):
                raise ValueError("Booleans not allowed")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Unsupported literal")

        # Backward compatibility for older Python versions
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            val = n.n
            if isinstance(val, bool):
                raise ValueError("Booleans not allowed")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Unsupported numeric literal")

        # Disallow names, calls, attributes, comprehensions, etc.
        raise ValueError("Unsupported expression node")

    try:
        return _eval(node)
    except Exception:
        return 'Execution Blocked!'
