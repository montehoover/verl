import ast

def evaluate_expression(expression):
    """
    Evaluate a simple arithmetic expression provided as a string.
    Supported operations: +, -, *, /, parentheses, and unary +/-
    Returns the result as a string.
    Raises ValueError for invalid expressions.
    """
    if not isinstance(expression, str):
        raise ValueError("Invalid expression")

    try:
        node = ast.parse(expression, mode='eval')
    except Exception:
        raise ValueError("Invalid expression")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)

            if not _is_number(left) or not _is_number(right):
                raise ValueError("Invalid expression")

            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right

            raise ValueError("Invalid expression")

        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if not _is_number(operand):
                raise ValueError("Invalid expression")

            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand

            raise ValueError("Invalid expression")

        # Support numeric literals
        if isinstance(n, ast.Constant):
            val = n.value
            if _is_number(val):
                return val
            raise ValueError("Invalid expression")

        # For older Python versions where numbers may be ast.Num
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            val = n.n  # type: ignore[attr-defined]
            if _is_number(val):
                return val
            raise ValueError("Invalid expression")

        raise ValueError("Invalid expression")

    def _is_number(x):
        # Exclude booleans (bool is subclass of int)
        return (isinstance(x, (int, float)) and not isinstance(x, bool))

    try:
        result = _eval(node)
    except Exception:
        raise ValueError("Invalid expression")

    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError("Invalid expression")

    if isinstance(result, float):
        if result.is_integer():
            return str(int(result))
        return format(result, ".15g")
    return str(result)
