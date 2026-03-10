import ast
from typing import Any


_ALLOWED_BINOPS = (
    ast.Add,  # +
    ast.Sub,  # -
    ast.Mult,  # *
    ast.Div,  # /
    ast.FloorDiv,  # //
    ast.Mod,  # %
    ast.Pow,  # **
)

_ALLOWED_UNARYOPS = (
    ast.UAdd,  # +
    ast.USub,  # -
    ast.Not,   # not
)

_ALLOWED_BOOLOPS = (
    ast.And,  # and
    ast.Or,   # or
)

_ALLOWED_CMPOPS = (
    ast.Eq,   # ==
    ast.NotEq,  # !=
    ast.Lt,   # <
    ast.LtE,  # <=
    ast.Gt,   # >
    ast.GtE,  # >=
)


def _is_safe_node(node: ast.AST) -> bool:
    # Entry node can be Expression in 'eval' mode, but we'll always pass body in parse_expression.
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, _ALLOWED_BINOPS)
            and _is_safe_node(node.left)
            and _is_safe_node(node.right)
        )

    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, _ALLOWED_UNARYOPS) and _is_safe_node(node.operand)

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _ALLOWED_BOOLOPS):
            return False
        return all(_is_safe_node(v) for v in node.values)

    if isinstance(node, ast.Compare):
        # Allow chained comparisons like 1 < x <= 3 (if names were allowed; here only constants)
        if not all(isinstance(op, _ALLOWED_CMPOPS) for op in node.ops):
            return False
        if not _is_safe_node(node.left):
            return False
        return all(_is_safe_node(comp) for comp in node.comparators)

    # Constants: allow numeric and boolean literals only
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, bool))

    # For older Python versions where booleans may appear as NameConstant
    if hasattr(ast, "NameConstant") and isinstance(node, getattr(ast, "NameConstant")):
        return node.value in (True, False)

    # Disallow tuples/lists/sets/dicts, calls, attributes, subscripts, comprehensions, lambdas, etc.
    # Explicitly block Name usage (variables), except possibly "True"/"False" which should be Constant in modern Python.
    if isinstance(node, ast.Name):
        return node.id in ("True", "False")

    # Parenthesized expressions do not create a special node; no explicit handling needed.

    return False


def parse_expression(expr: str) -> bool:
    """
    Validate a user-supplied expression for safety and allowed operations.

    Allowed:
      - Arithmetic: +, -, *, /, //, %, **
      - Unary: +x, -x, not x
      - Logical: and, or
      - Comparisons: ==, !=, <, <=, >, >=
      - Literals: integers, floats, booleans

    Disallowed:
      - Any names/variables (except boolean literals), function calls, attribute access,
        subscripts, comprehensions, lambdas, if-expressions, imports, etc.

    Returns True if the expression parses and only contains allowed operations and literals.
    """
    if not isinstance(expr, str):
        return False

    # Optional guard against extremely large inputs
    if len(expr) > 10000:
        return False

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    except Exception:
        return False

    try:
        return _is_safe_node(tree.body)
    except RecursionError:
        # Extremely deep/recursive structures are considered invalid
        return False
