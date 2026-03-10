import ast
import operator
from typing import Union

# Supported operators mapping
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        val = _eval_node(node.operand)
        return _UNARY_OPS[type(node.op)](val)

    # Python 3.8+: ast.Constant; Python <3.8: ast.Num
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n  # type: ignore[union-attr]

    # Any other node types are not supported
    raise ValueError("Unsupported expression")


def evaluate_user_math(math_expr: str):
    """
    Evaluate a basic arithmetic expression containing +, -, *, /.
    Returns the computed value on success, or the string "Processing Failed!" on error.
    """
    try:
        if not isinstance(math_expr, str):
            return "Processing Failed!"
        parsed = ast.parse(math_expr, mode="eval")
        result = _eval_node(parsed)
        return result
    except Exception:
        return "Processing Failed!"
