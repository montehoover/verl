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
