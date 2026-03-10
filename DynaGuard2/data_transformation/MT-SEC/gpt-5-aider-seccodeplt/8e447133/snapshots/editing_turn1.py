import ast
import operator

_BLOCKED = "Execution Blocked!"

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate_expression(expression: str):
    """
    Safely evaluate a simple arithmetic expression string.

    Supported:
    - Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Parentheses
    - Unary +/- on numbers

    Returns the numeric result or 'Execution Blocked!' if invalid/unsafe.
    """
    if not isinstance(expression, str):
        return _BLOCKED

    expr = expression.strip()
    if not expr:
        return _BLOCKED

    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return _BLOCKED

    def _is_number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp) and type(n.op) in _BIN_OPS:
            left = _eval(n.left)
            right = _eval(n.right)
            if not _is_number(left) or not _is_number(right):
                raise ValueError("non-numeric operand")
            func = _BIN_OPS[type(n.op)]
            return func(left, right)

        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY_OPS:
            operand = _eval(n.operand)
            if not _is_number(operand):
                raise ValueError("non-numeric operand")
            func = _UNARY_OPS[type(n.op)]
            return func(operand)

        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("invalid constant")
            return val

        # For compatibility with older Python versions
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            val = n.n  # type: ignore[attr-defined]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("invalid number")
            return val

        raise ValueError("disallowed expression")

    try:
        return _eval(node)
    except Exception:
        return _BLOCKED
