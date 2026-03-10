import ast
import operator as _op
from typing import Union

_Number = Union[int, float]

_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}

_UNARY_OPS = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
}


def evaluate_expression(expression: str) -> _Number:
    """
    Safely evaluate a simple arithmetic Python expression.

    Supported:
      - Numbers: int, float
      - Binary ops: +, -, *, /, //, %, **
      - Unary ops: +, -
      - Parentheses

    Disallowed:
      - Names, attribute access, calls, indexing, literals other than numbers.
    """
    if not isinstance(expression, str):
        raise TypeError("expression must be a string")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression") from e

    return _eval_node(tree.body)


def _eval_node(node) -> _Number:
    # Numeric literal (py3.8+: Constant; older: Num)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only integer and float literals are allowed")
    if isinstance(node, ast.Num):  # pragma: no cover - for very old Python
        if isinstance(node.n, (int, float)):
            return node.n
        raise ValueError("Only integer and float literals are allowed")

    # Parentheses are represented by the inner expression node directly.

    # Unary operations (+x, -x)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        operand = _eval_node(node.operand)
        return _UNARY_OPS[type(node.op)](operand)

    # Binary operations (x + y, x * y, etc.)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)

    # Anything else is disallowed
    raise ValueError(f"Disallowed expression: {type(node).__name__}")
