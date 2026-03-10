import ast
import math
from typing import Union

Number = Union[int, float]

# Limits to keep evaluation safe and bounded
_MAX_EXPR_LEN = 1000
_MAX_AST_NODES = 1000
_MAX_INT_DIGITS = 1000        # Maximum digits allowed in intermediate integer results
_MAX_FLOAT_ABS = 1e308        # Roughly the maximum finite double value
_MAX_POW_INT_BASE_ABS = 1_000_000
_MAX_POW_INT_EXP_ABS = 100
_MAX_POW_FLOAT_BASE_ABS = 1_000_000.0
_MAX_POW_FLOAT_EXP_ABS = 1_000.0


def perform_safe_math(expression: str) -> Number:
    """
    Evaluates a mathematical expression provided as a string in a safe manner.

    Args:
        expression: str - The mathematical expression to evaluate.

    Returns:
        The numerical result of the evaluated expression.

    Raises:
        ValueError: If the input is invalid, contains unsafe constructs, or is improperly formatted.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")
    expr = expression.strip()
    if not expr:
        raise ValueError("Expression cannot be empty or whitespace.")
    if len(expr) > _MAX_EXPR_LEN:
        raise ValueError("Expression is too long.")

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        raise ValueError("Invalid expression syntax.")

    # Limit expression complexity
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError("Expression is too complex.")

    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def _ensure_number(value: object) -> Number:
        if isinstance(value, bool):  # bool is a subclass of int; disallow explicitly
            raise ValueError("Booleans are not allowed.")
        if not isinstance(value, (int, float)):
            raise ValueError("Only numeric values are allowed.")
        return value

    def _check_limits(value: Number) -> Number:
        if isinstance(value, int):
            # Limit number of digits for big integers
            if value == 0:
                return value
            if len(str(abs(value))) > _MAX_INT_DIGITS:
                raise ValueError("Result exceeds allowed integer size.")
            return value
        else:
            # For floats, ensure finiteness and magnitude bounds
            if not math.isfinite(value):
                raise ValueError("Non-finite float result.")
            if abs(value) > _MAX_FLOAT_ABS:
                raise ValueError("Float result exceeds allowed magnitude.")
            return value

    def _safe_pow(left: Number, right: Number) -> Number:
        _ensure_number(left)
        _ensure_number(right)

        # If both are integers, constrain base and exponent tightly
        if isinstance(left, int) and isinstance(right, int):
            if abs(left) > _MAX_POW_INT_BASE_ABS:
                raise ValueError("Exponentiation base too large.")
            if abs(right) > _MAX_POW_INT_EXP_ABS:
                raise ValueError("Exponent too large.")
            result = left ** right
            return _check_limits(result)

        # For any float involvement, keep base/exponent within reasonable bounds
        if abs(float(left)) > _MAX_POW_FLOAT_BASE_ABS:
            raise ValueError("Exponentiation base too large.")
        if abs(float(right)) > _MAX_POW_FLOAT_EXP_ABS:
            raise ValueError("Exponent too large.")

        try:
            result = left ** right
        except OverflowError:
            raise ValueError("Exponentiation overflow.")
        return _check_limits(result)

    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numeric literals
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("Only numeric constants are allowed.")
            return _ensure_number(val)

        # Compatibility with older Python versions where ast.Num is used
        if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
            val = node.n
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("Only numeric constants are allowed.")
            return _ensure_number(val)

        # Unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unary_ops):
                raise ValueError("Unsupported unary operator.")
            operand = _ensure_number(_eval(node.operand))
            if isinstance(node.op, ast.UAdd):
                return _check_limits(+operand)
            elif isinstance(node.op, ast.USub):
                return _check_limits(-operand)

        # Binary operations
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_bin_ops):
                raise ValueError("Unsupported binary operator.")
            left = _ensure_number(_eval(node.left))
            right = _ensure_number(_eval(node.right))

            if isinstance(node.op, ast.Add):
                return _check_limits(left + right)
            if isinstance(node.op, ast.Sub):
                return _check_limits(left - right)
            if isinstance(node.op, ast.Mult):
                return _check_limits(left * right)
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("Division by zero.")
                return _check_limits(left / right)
            if isinstance(node.op, ast.FloorDiv):
                if right == 0:
                    raise ValueError("Division by zero.")
                return _check_limits(left // right)
            if isinstance(node.op, ast.Mod):
                if right == 0:
                    raise ValueError("Modulo by zero.")
                return _check_limits(left % right)
            if isinstance(node.op, ast.Pow):
                return _safe_pow(left, right)

        # Disallow everything else explicitly
        raise ValueError("Unsupported or unsafe expression element encountered.")

    result = _eval(tree)
    return result
