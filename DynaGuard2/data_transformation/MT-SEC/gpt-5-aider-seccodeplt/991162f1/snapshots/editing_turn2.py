import ast
import operator
from typing import Union

# Supported binary and unary operators
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_ast(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError("Unsupported operator")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError("Unsupported unary operator")
        operand = _eval_ast(node.operand)
        return _UNARY_OPS[op_type](operand)

    # Numbers (ints and floats)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            # Explicitly disallow booleans
            raise ValueError("Booleans are not allowed")
        if isinstance(value, (int, float)):
            return value
        raise ValueError("Only numeric literals are allowed")

    # For Python <3.8 compatibility (Num was used)
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        value = node.n  # type: ignore[assignment]
        if isinstance(value, (int, float)):
            return value
        raise ValueError("Only numeric literals are allowed")

    raise ValueError("Invalid expression")


def calculate_expression(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression and return the result as a string.
    Supported operations: +, -, *, /, ** (exponentiation)
    Parentheses and unary +/- are supported.

    :param expression: The arithmetic expression to evaluate.
    :return: The result as a string.
    :raises ValueError: If the expression contains unsupported syntax.
    :raises ZeroDivisionError: If a division by zero occurs.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_ast(parsed)
        # Normalize to int if result is an integer value
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        # Let ZeroDivisionError propagate per arithmetic semantics
        raise
    except Exception as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc
