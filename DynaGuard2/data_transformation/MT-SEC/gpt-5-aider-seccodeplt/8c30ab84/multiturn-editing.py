import ast
from typing import Union

Number = Union[int, float]


def _is_number(value) -> bool:
    # Accept ints and floats but reject booleans
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_eval(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right

        raise ValueError("Unsupported operator in expression.")

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator in expression.")

    if isinstance(node, ast.Constant):
        if _is_number(node.value):
            return node.value
        raise ValueError("Unsupported constant in expression.")

    # For Python versions < 3.8 where numbers appear as ast.Num
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        if _is_number(node.n):  # type: ignore[attr-defined]
            return node.n  # type: ignore[attr-defined]
        raise ValueError("Unsupported number in expression.")

    # Reject everything else (Names, Calls, Attributes, etc.)
    raise ValueError("Unsafe or invalid expression.")


def compute_expression(expression_input: str) -> Number:
    """
    Safely evaluate a mathematical expression string and return its numeric result.

    Allowed:
    - Numbers (integers and floats)
    - Binary operators: +, -, *, /, %, **
    - Unary operators: +, -
    - Parentheses

    Raises ValueError for any unsafe or invalid input.
    """
    if not isinstance(expression_input, str):
        raise ValueError("Input must be a string.")

    expr = expression_input.strip()
    if not expr:
        raise ValueError("Expression cannot be empty.")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression.") from e

    try:
        return _safe_eval(tree)
    except ZeroDivisionError as e:
        raise ValueError("Division by zero.") from e
    except ValueError:
        # Propagate validation errors as-is
        raise
