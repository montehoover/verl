import ast

_BLOCKED = "Execution Blocked!"


_UNARY_OPS = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
}


def calculate_expression(expression: str):
    """
    Safely evaluate a simple arithmetic or string expression string.

    Supported:
    - Numbers: Addition (+), Subtraction (-), Multiplication (*), Division (/)
    - Strings: Concatenation with + (e.g., 'a' + 'b')
               Repetition with * when combined with an integer (e.g., 'a' * 3 or 3 * 'a')
    - Parentheses
    - Unary +/- on numbers

    Returns the result (number or string) or 'Execution Blocked!' if invalid/unsafe.
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

    def _is_int(x):
        return isinstance(x, int) and not isinstance(x, bool)

    def _is_string(x):
        return isinstance(x, str)

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)

            # Addition: numbers or strings
            if isinstance(n.op, ast.Add):
                if _is_number(left) and _is_number(right):
                    return left + right
                if _is_string(left) and _is_string(right):
                    return left + right
                raise ValueError("invalid operands for +")

            # Subtraction: numbers only
            if isinstance(n.op, ast.Sub):
                if _is_number(left) and _is_number(right):
                    return left - right
                raise ValueError("invalid operands for -")

            # Multiplication: numbers*numbers, or string*int / int*string
            if isinstance(n.op, ast.Mult):
                if _is_number(left) and _is_number(right):
                    return left * right
                if _is_string(left) and _is_int(right):
                    return left * right
                if _is_string(right) and _is_int(left):
                    return left * right
                raise ValueError("invalid operands for *")

            # Division: numbers only (true division)
            if isinstance(n.op, ast.Div):
                if _is_number(left) and _is_number(right):
                    return left / right
                raise ValueError("invalid operands for /")

            # Any other binary operators are disallowed
            raise ValueError("disallowed binary operator")

        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY_OPS:
            operand = _eval(n.operand)
            if not _is_number(operand):
                raise ValueError("non-numeric operand for unary op")
            func = _UNARY_OPS[type(n.op)]
            return func(operand)

        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, bool) or not isinstance(val, (int, float, str)):
                raise ValueError("invalid constant")
            return val

        # For compatibility with older Python versions
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            val = n.n  # type: ignore[attr-defined]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("invalid number")
            return val

        if hasattr(ast, "Str") and isinstance(n, ast.Str):  # type: ignore[attr-defined]
            val = n.s  # type: ignore[attr-defined]
            if not isinstance(val, str):
                raise ValueError("invalid string")
            return val

        raise ValueError("disallowed expression")

    try:
        return _eval(node)
    except Exception:
        return _BLOCKED


def execute_user_input(user_code: str):
    """
    Securely execute user-provided code that should be a simple expression
    supporting basic arithmetic and string operations (same as calculate_expression).
    Returns the evaluated result or 'Execution Blocked!' if invalid/unsafe.
    """
    return calculate_expression(user_code)
