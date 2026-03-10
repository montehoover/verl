import ast
import math

def evaluate_expression(expression, variables=None):
    """
    Evaluate a simple arithmetic expression provided as a string with optional variables.
    Supported operations: +, -, *, /, parentheses, and unary +/-
    Variables can be supplied via the `variables` dict mapping names to numeric values.
    Returns the natural logarithm of the evaluated result as a string.
    Raises ValueError for invalid expressions or if log(result) is undefined.
    """
    if not isinstance(expression, str):
        raise ValueError("Invalid expression")

    if variables is not None and not isinstance(variables, dict):
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

        if isinstance(n, ast.Name):
            if variables is None or n.id not in variables:
                raise ValueError("Invalid expression")
            val = variables[n.id]
            if not _is_number(val):
                raise ValueError("Invalid expression")
            return val

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

    # Compute natural logarithm of the result; must be positive and finite
    try:
        if not (isinstance(result, (int, float)) and result > 0):
            raise ValueError("Invalid expression")
        log_result = math.log(result)
        if not math.isfinite(log_result):
            raise ValueError("Invalid expression")
    except Exception:
        raise ValueError("Invalid expression")

    if isinstance(log_result, float):
        if log_result.is_integer():
            return str(int(log_result))
        return format(log_result, ".15g")
    return str(log_result)
