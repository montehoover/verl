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
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_BOOL_OPS = {ast.And, ast.Or}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _eval_node(node: ast.AST) -> Union[int, float, bool]:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        # Guard against extremely large exponents in power operation
        if isinstance(node.op, ast.Pow):
            # Allow only reasonably small exponents to avoid resource exhaustion
            if isinstance(right, (int, float)) and abs(right) > 100:
                raise ValueError("Exponent too large")

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

    if isinstance(node, ast.Compare):
        # Handle chained comparisons: a < b < c
        left_val = _eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _CMP_OPS:
                raise ValueError("Unsupported comparison operator")
            right_val = _eval_node(comparator)
            if not _CMP_OPS[type(op)](left_val, right_val):
                return False
            left_val = right_val
        return True

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


def evaluate_user_math(math_expr: str) -> str:
    """
    Safely evaluate a user-provided expression containing arithmetic and logical operations.
    Supports: +, -, *, /, //, %, **, unary +/-, comparisons (==, !=, <, <=, >, >=),
    and logical operators AND, OR, NOT (case-insensitive). Returns the result as a string
    on success, or "Processing Failed!" if evaluation fails or encounters unsupported input.
    """
    try:
        if not isinstance(math_expr, str):
            return "Processing Failed!"
        normalized = _normalize_expr(math_expr)
        # Parse in 'eval' mode to disallow statements (e.g., assignments, imports)
        parsed = ast.parse(normalized, mode="eval")
        result = _eval_node(parsed)
        return str(result)
    except Exception:
        return "Processing Failed!"
