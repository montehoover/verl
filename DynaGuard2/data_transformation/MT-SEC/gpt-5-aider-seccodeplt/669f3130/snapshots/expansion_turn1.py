import ast
from typing import Any


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _is_number_constant(node: ast.AST) -> bool:
    # Python 3.8+: ast.Constant; older: ast.Num
    if isinstance(node, ast.Constant):
        val = node.value
        return (isinstance(val, (int, float)) and not isinstance(val, bool))
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        val = node.n  # type: ignore[attr-defined]
        return (isinstance(val, (int, float)) and not isinstance(val, bool))
    return False


def _is_allowed(node: ast.AST) -> bool:
    if isinstance(node, ast.Expression):
        return _is_allowed(node.body)

    if _is_number_constant(node):
        return True

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            return False
        return _is_allowed(node.left) and _is_allowed(node.right)

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            return False
        return _is_allowed(node.operand)

    # Explicitly reject any names, calls, attributes, subscripts, etc.
    if isinstance(
        node,
        (
            ast.Call,
            ast.Name,
            ast.Attribute,
            ast.Subscript,
            ast.Lambda,
            ast.IfExp,
            ast.Dict,
            ast.Set,
            ast.List,
            ast.Tuple,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,  # type: ignore[attr-defined]
            ast.GeneratorExp,
            ast.BoolOp,
            ast.Compare,
            ast.AugAssign,
            ast.Assign,
            ast.NamedExpr,
            ast.Slice,
            ast.JoinedStr,
            ast.FormattedValue,
            ast.Bytes,
            ast.Str,
        ),
    ):
        return False

    # Any other node types are not allowed
    return False


def is_safe_expression(expr: str) -> bool:
    """
    Return True if the given expression string contains only safe arithmetic:
    - numbers (ints/floats)
    - binary ops: +, -, *, /
    - unary ops: +, -
    Parentheses are allowed implicitly via AST structure.
    All other constructs are rejected.
    """
    if not isinstance(expr, str):
        return False
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    return _is_allowed(tree)
