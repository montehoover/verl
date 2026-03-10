import ast
import math


def compute_expression(expr: str):
    """
    Evaluate a mathematical expression safely and return the result.

    Args:
        expr (str): A string containing a mathematical expression.

    Returns:
        int | float: The evaluated result of the expression.

    Raises:
        ValueError: If the expression includes unsupported operators/characters,
        contains unsafe code, or if evaluation fails.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")
    if expr.strip() == "":
        raise ValueError("Empty expression.")

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        raise ValueError("Invalid expression syntax.")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)

            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Operands must be numeric.")

            op = node.op
            if isinstance(op, ast.Pow):
                if isinstance(right, (int, float)) and abs(right) > 10000:
                    raise ValueError("Exponent too large.")
                return left ** right
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.FloorDiv):
                return left // right
            if isinstance(op, ast.Mod):
                return left % right
            raise ValueError("Unsupported operator.")

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if not isinstance(operand, (int, float)):
                raise ValueError("Operand must be numeric.")
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")

        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Unsupported constant.")
            return value

        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            value = node.n
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Unsupported number literal.")
            return value

        raise ValueError("Unsupported expression element.")

    try:
        result = _eval(tree)
    except ZeroDivisionError:
        raise ValueError("Division by zero.") from None
    except RecursionError:
        raise ValueError("Expression too complex.") from None
    except ValueError:
        raise
    except Exception:
        raise ValueError("Failed to evaluate expression.") from None

    if not isinstance(result, (int, float)):
        raise ValueError("Unsupported result type.")
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("Non-finite result.")

    return result
