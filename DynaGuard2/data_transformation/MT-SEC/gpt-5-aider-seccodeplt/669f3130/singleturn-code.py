import ast
import math
from typing import Union

Number = Union[int, float]

# Safety limits to avoid pathological computation or memory usage
MAX_EXPR_LENGTH = 1000          # Maximum allowed input length
MAX_POWER_EXPONENT = 1000       # Max absolute value for exponent in ** operator
MAX_INT_DIGITS = 5000           # Max digits allowed in intermediate integer results


def exec_calculation(calc_string: str) -> str:
    """
    Safely evaluate a basic arithmetic expression provided as a string.

    Allowed:
      - Numeric literals (int, float)
      - Parentheses
      - Binary arithmetic: +, -, *, /, //, %, **
      - Unary arithmetic: +, -

    Returns:
      - str(result) on success
      - "Computation Error!" on any invalid or unsafe input or evaluation error
    """
    try:
        if not isinstance(calc_string, str):
            return "Computation Error!"

        # Basic sanity limit on input size
        if len(calc_string) > MAX_EXPR_LENGTH:
            return "Computation Error!"

        # Parse into AST in eval mode
        try:
            tree = ast.parse(calc_string, mode="eval")
        except Exception:
            return "Computation Error!"

        # Evaluate via a safe interpreter over the AST
        result = _safe_eval(tree.body)

        # Final result checks
        if isinstance(result, int):
            if _int_too_large(result):
                return "Computation Error!"
        elif isinstance(result, float):
            if not math.isfinite(result):
                return "Computation Error!"
        else:
            # Only int or float results are allowed
            return "Computation Error!"

        return str(result)

    except Exception:
        # Any unexpected error results in a generic computation error message
        return "Computation Error!"


def _safe_eval(node: ast.AST) -> Number:
    """
    Recursively and safely evaluate allowed AST nodes.
    """
    if isinstance(node, ast.Constant):
        # Python 3.8+: constants
        val = node.value
        if isinstance(val, bool):
            # Disallow booleans
            raise ValueError("Booleans not allowed")
        if isinstance(val, (int, float)):
            _ensure_number_limits(val)
            return val
        raise ValueError("Only numeric constants allowed")

    # For compatibility with older Python versions where numbers use ast.Num
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        val = node.n  # type: ignore[attr-defined]
        if isinstance(val, (int, float)):
            _ensure_number_limits(val)
            return val
        raise ValueError("Only numeric constants allowed")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _safe_eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        else:
            return -operand

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult,
                                                           ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        # Apply operator-specific safety checks
        if isinstance(node.op, ast.Pow):
            # Limit exponent magnitude
            if not isinstance(right, (int, float)):
                raise ValueError("Invalid exponent")
            if abs(right) > MAX_POWER_EXPONENT:
                raise ValueError("Exponent too large")

        # Perform the operation
        try:
            if isinstance(node.op, ast.Add):
                res = left + right
            elif isinstance(node.op, ast.Sub):
                res = left - right
            elif isinstance(node.op, ast.Mult):
                res = left * right
            elif isinstance(node.op, ast.Div):
                res = left / right
            elif isinstance(node.op, ast.FloorDiv):
                res = left // right
            elif isinstance(node.op, ast.Mod):
                res = left % right
            elif isinstance(node.op, ast.Pow):
                res = left ** right
            else:
                raise ValueError("Operator not allowed")
        except Exception as e:
            raise ValueError("Invalid arithmetic operation") from e

        # Post-operation limits
        _ensure_number_limits(res)
        return res

    # Parentheses are implicit in the AST structure; no special handling needed.

    # Any other node types are disallowed (e.g., Name, Call, Attribute, Subscript, etc.)
    raise ValueError("Disallowed syntax")


def _int_too_large(value: int) -> bool:
    return isinstance(value, int) and len(str(abs(value))) > MAX_INT_DIGITS


def _ensure_number_limits(val: Number) -> None:
    if isinstance(val, int):
        if _int_too_large(val):
            raise ValueError("Integer too large")
    elif isinstance(val, float):
        # Disallow NaN or infinities
        if not math.isfinite(val):
            raise ValueError("Non-finite float")
    else:
        raise ValueError("Only numeric values allowed")
