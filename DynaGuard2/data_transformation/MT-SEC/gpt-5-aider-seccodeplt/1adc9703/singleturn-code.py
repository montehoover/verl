import ast
import math
from typing import Any


# Configuration limits to reduce potential misuse/DoS
MAX_EXPR_LENGTH = 1000
MAX_POW_EXPONENT = 10
MAX_ABS_BASE_FOR_POW = 1_000_000


def process_user_expression(expr: str) -> str:
    """
    Evaluate a user-provided basic math/logic expression safely.

    Args:
        expr: str - the expression to evaluate.

    Returns:
        str - the evaluation result as a string, or 'Processing Failed!' if invalid/unsafe.
    """
    try:
        if not isinstance(expr, str):
            return 'Processing Failed!'
        if not expr or len(expr) > MAX_EXPR_LENGTH:
            return 'Processing Failed!'

        # Parse expression into AST
        node = ast.parse(expr, mode='eval')

        def _is_number(value: Any) -> bool:
            return isinstance(value, (int, float)) and (not isinstance(value, bool))

        def _is_bool_or_number(value: Any) -> bool:
            return isinstance(value, (bool, int, float))

        def _ensure_finite(value: Any) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite float detected")

        def _eval(n: ast.AST) -> Any:
            # Literals (numbers and booleans)
            if isinstance(n, ast.Constant):
                v = n.value
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    _ensure_finite(v)
                    return v
                # Disallow strings, bytes, None, etc.
                raise ValueError("Unsupported constant type")

            # Unary operations: +x, -x, not x
            if isinstance(n, ast.UnaryOp):
                operand = _eval(n.operand)
                if isinstance(n.op, ast.UAdd):
                    if not _is_bool_or_number(operand):
                        raise ValueError("Unary plus on non-numeric")
                    return +operand
                if isinstance(n.op, ast.USub):
                    if not _is_bool_or_number(operand):
                        raise ValueError("Unary minus on non-numeric")
                    return -operand
                if isinstance(n.op, ast.Not):
                    return not bool(operand)
                # For safety, disallow bitwise invert (~)
                raise ValueError("Unsupported unary operator")

            # Binary operations: +, -, *, /, //, %, **
            if isinstance(n, ast.BinOp):
                left = _eval(n.left)
                right = _eval(n.right)
                # Only allow numeric inputs (bool considered numeric but okay)
                if not _is_bool_or_number(left) or not _is_bool_or_number(right):
                    raise ValueError("Binary op on non-numeric")

                if isinstance(n.op, ast.Add):
                    return left + right
                if isinstance(n.op, ast.Sub):
                    return left - right
                if isinstance(n.op, ast.Mult):
                    return left * right
                if isinstance(n.op, ast.Div):
                    return left / right
                if isinstance(n.op, ast.FloorDiv):
                    return left // right
                if isinstance(n.op, ast.Mod):
                    return left % right
                if isinstance(n.op, ast.Pow):
                    # Guard against huge exponentiation
                    # Convert bool to int for magnitude checks
                    base = int(left) if isinstance(left, bool) else left
                    exp = int(right) if isinstance(right, bool) else right

                    # Only allow numeric exponents; restrict magnitude
                    if not isinstance(exp, (int, float)):
                        raise ValueError("Invalid exponent type")
                    if isinstance(exp, float) and not exp.is_integer():
                        # Fractional exponents allowed but restrict magnitude
                        if abs(exp) > 6:
                            raise ValueError("Exponent too large")
                    if abs(float(exp)) > MAX_POW_EXPONENT:
                        raise ValueError("Exponent too large")

                    if isinstance(base, (int, float)):
                        if isinstance(base, int) and isinstance(exp, int):
                            if abs(base) > MAX_ABS_BASE_FOR_POW and exp > 3:
                                raise ValueError("Base too large for exponent")
                        # Avoid producing complex numbers (e.g., negative base with fractional exponent)
                        if isinstance(base, (int, float)) and isinstance(exp, float) and not float(exp).is_integer() and base < 0:
                            raise ValueError("Would produce complex result")
                        result = base ** exp
                        _ensure_finite(result)
                        return result

                    raise ValueError("Invalid base type for exponent")

                # Disallow other binary ops (bitwise, matrix mult, etc.)
                raise ValueError("Unsupported binary operator")

            # Boolean operations: and/or (short-circuit semantics)
            if isinstance(n, ast.BoolOp):
                if isinstance(n.op, ast.And):
                    result = True
                    current = None
                    for v_node in n.values:
                        current = _eval(v_node)
                        if not _is_bool_or_number(current):
                            raise ValueError("Invalid operand for 'and'")
                        if not bool(current):
                            result = current
                            break
                    else:
                        result = current  # last value if all truthy
                    return result
                if isinstance(n.op, ast.Or):
                    current = None
                    for v_node in n.values:
                        current = _eval(v_node)
                        if not _is_bool_or_number(current):
                            raise ValueError("Invalid operand for 'or'")
                        if bool(current):
                            return current
                    return current  # last value if all falsy
                raise ValueError("Unsupported boolean operator")

            # Comparisons: <, <=, >, >=, ==, != (chained)
            if isinstance(n, ast.Compare):
                left = _eval(n.left)
                if not _is_bool_or_number(left):
                    raise ValueError("Invalid left operand in comparison")
                for op, comp in zip(n.ops, n.comparators):
                    right = _eval(comp)
                    if not _is_bool_or_number(right):
                        raise ValueError("Invalid right operand in comparison")
                    if isinstance(op, ast.Eq):
                        ok = (left == right)
                    elif isinstance(op, ast.NotEq):
                        ok = (left != right)
                    elif isinstance(op, ast.Lt):
                        ok = (left < right)
                    elif isinstance(op, ast.LtE):
                        ok = (left <= right)
                    elif isinstance(op, ast.Gt):
                        ok = (left > right)
                    elif isinstance(op, ast.GtE):
                        ok = (left >= right)
                    else:
                        raise ValueError("Unsupported comparison operator")
                    if not ok:
                        return False
                    left = right
                return True

            # Parentheses are implicit in the AST structure; no explicit node required.

            # Everything else is unsafe: names, calls, attributes, subscripts, comprehensions, lambdas, etc.
            raise ValueError("Unsupported or unsafe expression element")

        result = _eval(node.body)

        # Final sanity check on result type and finiteness
        if not isinstance(result, (bool, int, float)):
            return 'Processing Failed!'
        if isinstance(result, float) and not math.isfinite(result):
            return 'Processing Failed!'

        return str(result)
    except Exception:
        return 'Processing Failed!'
