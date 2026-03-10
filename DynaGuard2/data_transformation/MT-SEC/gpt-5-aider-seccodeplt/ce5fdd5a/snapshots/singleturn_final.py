import ast
import math
from typing import Any

# Safety limits
_MAX_EXPR_LEN = 1000
_MAX_AST_NODES = 200
_MAX_ABS_INT = 10 ** 100  # cap for intermediate/result integer magnitude
_MAX_FLOAT_ABS = 1e308
_MAX_POWER_EXP = 1000
_MAX_POWER_BASE_ABS = 10 ** 6


def evaluate_expression(expr: str) -> str:
    """
    Safely evaluate a simple arithmetic Python expression provided as a string.

    Allowed:
      - Literals: ints, floats
      - Unary ops: +, -
      - Binary ops: +, -, *, /, //, %, **

    Disallowed:
      - Names, calls, attributes, subscripts, comprehensions, lambdas, etc.

    Returns:
      - str(result) on success
      - 'Error!' if input is untrustworthy or any exception occurs
    """
    try:
        if not isinstance(expr, str):
            return "Error!"
        # Basic sanity checks
        if len(expr) == 0 or len(expr) > _MAX_EXPR_LEN:
            return "Error!"

        # Parse to AST
        try:
            tree = ast.parse(expr, mode="eval")
        except Exception:
            return "Error!"

        # Evaluate with strict whitelist
        node_counter = {"count": 0}

        def _count_node():
            node_counter["count"] += 1
            if node_counter["count"] > _MAX_AST_NODES:
                raise ValueError("Too many nodes")

        def _is_number(x: Any) -> bool:
            # Exclude bool, which is a subclass of int
            return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, float)

        def _check_value(v: Any) -> None:
            if isinstance(v, int) and not isinstance(v, bool):
                if abs(v) > _MAX_ABS_INT:
                    raise ValueError("Integer magnitude too large")
            elif isinstance(v, float):
                if not math.isfinite(v) or abs(v) > _MAX_FLOAT_ABS:
                    raise ValueError("Float out of range")
            else:
                raise ValueError("Invalid result type")

        def _eval(node: ast.AST) -> Any:
            _count_node()

            if isinstance(node, ast.Expression):
                return _eval(node.body)

            # Constants (numbers)
            if isinstance(node, ast.Constant):
                val = node.value
                # Only allow ints/floats (no bools, strings, complex, etc.)
                if isinstance(val, bool):
                    raise ValueError("Booleans not allowed")
                if isinstance(val, (int, float)):
                    if isinstance(val, float):
                        if not math.isfinite(val):
                            raise ValueError("Non-finite float")
                        if abs(val) > _MAX_FLOAT_ABS:
                            raise ValueError("Float too large")
                    else:
                        if abs(val) > _MAX_ABS_INT:
                            raise ValueError("Integer too large")
                    return val
                raise ValueError("Unsupported literal")

            # Unary operations: +, -
            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, (ast.UAdd, ast.USub)):
                    operand = _eval(node.operand)
                    if not _is_number(operand):
                        raise ValueError("Unary op on non-number")
                    res = +operand if isinstance(node.op, ast.UAdd) else -operand
                    _check_value(res)
                    return res
                raise ValueError("Unsupported unary operator")

            # Binary operations: +, -, *, /, //, %, **
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                if not (_is_number(left) and _is_number(right)):
                    raise ValueError("Binary op on non-number")

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
                        # Restrict exponent and base magnitude to prevent abuse
                        if not isinstance(right, int) or isinstance(right, bool):
                            raise ValueError("Exponent must be a non-negative integer")
                        if right < 0 or right > _MAX_POWER_EXP:
                            raise ValueError("Exponent out of range")
                        if isinstance(left, (int, float)) and abs(left) > _MAX_POWER_BASE_ABS:
                            raise ValueError("Base out of range")
                        res = left ** right
                    else:
                        raise ValueError("Unsupported binary operator")
                except Exception as e:
                    raise ValueError(str(e))

                _check_value(res)
                return res

            # Parentheses are implicit in AST structure; no explicit handling needed.

            # Disallow everything else
            raise ValueError("Unsupported syntax")

        result = _eval(tree)
        # Final type check and stringify
        if not isinstance(result, (int, float)) or isinstance(result, bool):
            return "Error!"
        return str(result)
    except Exception:
        return "Error!"
