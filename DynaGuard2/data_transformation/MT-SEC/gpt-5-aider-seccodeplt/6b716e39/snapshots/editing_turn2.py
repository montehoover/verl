import ast
import operator
import re
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
    ast.Not: operator.not_,
}

_BOOL_OPS = {ast.And, ast.Or}


def _eval_node(node: ast.AST) -> Union[int, float, bool]:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)

    if isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_OPS:
        # Short-circuit evaluation
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not bool(_eval_node(v)):
                    return False
            return True
        elif isinstance(node.op, ast.Or):
            for v in node.values:
                if bool(_eval_node(v)):
                    return True
            return False

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        val = _eval_node(node.operand)
        return _UNARY_OPS[type(node.op)](val)

    # Python 3.8+: ast.Constant; Python <3.8: ast.Num, ast.NameConstant
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n  # type: ignore[union-attr]
    if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant):  # type: ignore[attr-defined]
        if isinstance(node.value, bool):
            return node.value

    # Any other node types are not supported
    raise ValueError("Unsupported expression")


def _normalize_expr(expr: str) -> str:
    # Normalize logical operators and booleans (case-insensitive)
    replacements = [
        (r"\bAND\b", "and"),
        (r"\bOR\b", "or"),
        (r"\bNOT\b", "not"),
        (r"\bTRUE\b", "True"),
        (r"\bFALSE\b", "False"),
    ]
    normalized = expr
    for pattern, repl in replacements:
        normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)
    return normalized


def evaluate_user_math(math_expr: str):
    """
    Evaluate a basic expression containing arithmetic (+, -, *, /) and logical (AND, OR, NOT) operators.
    Returns the computed value on success (number or boolean), or the string "Processing Failed!" on error.
    """
    try:
        if not isinstance(math_expr, str):
            return "Processing Failed!"
        normalized = _normalize_expr(math_expr)
        parsed = ast.parse(normalized, mode="eval")
        result = _eval_node(parsed)
        return result
    except Exception:
        return "Processing Failed!"
