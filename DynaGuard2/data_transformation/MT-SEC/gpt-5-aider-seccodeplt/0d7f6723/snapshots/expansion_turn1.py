import ast
from typing import Union


_ALLOWED_NUMERIC_TYPES = (int, float)


def _is_allowed_expr(node: ast.AST) -> bool:
    # Literals
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (_ALLOWED_NUMERIC_TYPES + (str,)))
    # Binary operations: arithmetic and string concatenation/repetition
    if isinstance(node, ast.BinOp):
        if not isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ):
            return False
        return _is_allowed_expr(node.left) and _is_allowed_expr(node.right)
    # Unary operations: +x, -x
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            return False
        return _is_allowed_expr(node.operand)
    # Parentheses do not create nodes; tuples (with commas) should be disallowed
    if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return False
    # Disallow everything that could be harmful or evaluate external code/IO
    if isinstance(
        node,
        (
            ast.Call,
            ast.Attribute,
            ast.Subscript,
            ast.Name,
            ast.Lambda,
            ast.IfExp,
            ast.Compare,
            ast.BoolOp,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.Starred,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.JoinedStr,       # f-strings
            ast.FormattedValue,  # f-strings
        ),
    ):
        return False
    # Unknown/unsupported node type
    return False


def check_script_syntax(script: str) -> bool:
    """
    Returns True if the provided script contains only simple arithmetic
    (using numeric literals with +, -, *, /, //, %, **, unary +/-)
    or simple string operations (string literals with + for concatenation
    or * for repetition), and contains no names, calls, attributes, imports,
    or any other potentially harmful constructs.
    """
    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError:
        return False

    # Allow empty script
    if not tree.body:
        return True

    # Only allow expression statements where the expressions pass the whitelist
    for stmt in tree.body:
        # Disallow any non-expression statements (e.g., assignments, imports, etc.)
        if not isinstance(stmt, ast.Expr):
            return False
        if not _is_allowed_expr(stmt.value):
            return False

    return True
