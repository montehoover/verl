import ast
from typing import Any


def parse_user_expression(expr: str) -> bool:
    """
    Return True if expr is a valid, safe Python arithmetic expression consisting only of:
    - numeric literals (int, float)
    - parentheses
    - unary + and -
    - binary +, -, *, /, //, %, **

    No names, function calls, attributes, subscripts, comprehensions, or other operators are allowed.
    """
    if expr is None:
        return False

    expr = expr.strip()
    if not expr:
        return False

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    # Allowed node and operator types
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,  # Python 3.8+
        ast.Num,       # For compatibility with older Python versions
    )
    allowed_bin_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )
    allowed_unary_ops = (
        ast.UAdd,
        ast.USub,
    )

    for node in ast.walk(tree):
        # Allow only specific node types
        if isinstance(node, allowed_nodes):
            # Validate constants are only int or float (not bool, str, complex, etc.)
            if isinstance(node, ast.Constant):
                val = node.value
                if isinstance(val, bool):
                    return False
                if not isinstance(val, (int, float)):
                    return False
            elif isinstance(node, ast.Num):
                # ast.Num.n may be int, float, or complex; disallow complex
                val: Any = getattr(node, "n", None)
                if isinstance(val, bool):
                    return False
                if not isinstance(val, (int, float)):
                    return False
            continue

        # Allow specific operator nodes
        if isinstance(node, allowed_bin_ops):
            continue
        if isinstance(node, allowed_unary_ops):
            continue

        # Everything else is disallowed
        return False

    return True


def evaluate_safe_expression(expr: str) -> Any:
    """
    Safely evaluate a validated arithmetic expression.
    Returns the numeric result, or a string with an error message if evaluation fails or is unsafe.
    """
    if not parse_user_expression(expr):
        return "Error: Invalid or unsafe expression"

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"Error: Invalid syntax ({e})"

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return val
            raise ValueError("Unsupported constant type")
        if isinstance(node, ast.Num):
            val = getattr(node, "n", None)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return val
            raise ValueError("Unsupported numeric literal")

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if not isinstance(operand, (int, float)):
                raise ValueError("Operand must be a number")
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Operands must be numbers")

            op = node.op
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
            if isinstance(op, ast.Pow):
                return left ** right

            raise ValueError("Unsupported binary operator")

        # Any other node type is disallowed
        raise ValueError("Unsupported syntax")

    try:
        return _eval(tree.body)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except OverflowError:
        return "Error: Numerical overflow"
    except MemoryError:
        return "Error: Evaluation ran out of memory"
    except ValueError as e:
        return f"Error: {e}"
    except Exception:
        return "Error: Failed to evaluate expression"
