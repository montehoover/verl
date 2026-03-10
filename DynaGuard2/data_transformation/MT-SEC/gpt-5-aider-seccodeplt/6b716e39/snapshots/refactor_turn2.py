import ast
from typing import Any


def _fail() -> str:
    return 'Processing Failed!'


def _basic_checks(math_expr: str) -> str:
    if not isinstance(math_expr, str):
        raise ValueError("Not a string")
    expr = math_expr.strip()
    if not expr:
        raise ValueError("Empty expression")
    if len(expr) > 1000:
        raise ValueError("Expression too long")
    return expr


def _parse_to_ast(expr: str) -> ast.AST:
    # Parse expression in eval mode to ensure a single expression
    return ast.parse(expr, mode='eval')


def _security_check(tree: ast.AST) -> None:
    # Define allowed node and operator types
    allowed_nodes = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )

    # Older Python compatibility (Num node)
    if hasattr(ast, "Num"):
        allowed_nodes = allowed_nodes + (ast.Num,)  # type: ignore[attr-defined]

    for node in ast.walk(tree):
        # Reject any node type not in the allowed set
        if not isinstance(node, allowed_nodes):
            raise ValueError("Disallowed AST node")

        # Validate constants: allow only bool, int, float
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (bool, int, float)):
                raise ValueError("Disallowed constant type")

        # Validate legacy Num nodes if present
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            # ast.Num only holds numeric literals
            pass

        # Validate names: only True/False
        if isinstance(node, ast.Name):
            if node.id not in ("True", "False"):
                raise ValueError("Disallowed name")

        # Validate unary operators: only +, -, not
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
                raise ValueError("Disallowed unary operator")

        # Validate binary operators: allow +, -, *, /, //, %
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
                raise ValueError("Disallowed binary operator")

        # Validate boolean operators: and, or
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                raise ValueError("Disallowed boolean operator")

        # Validate comparisons: ==, !=, <, <=, >, >= only
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    raise ValueError("Disallowed comparison operator")


def _numify(val: Any) -> float | int:
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return val
    raise ValueError("Non-numeric value")


def _evaluate(node: ast.AST) -> Any:
    # Evaluate a safe AST produced and validated by _security_check
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        return node.value

    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n  # type: ignore[attr-defined]

    if isinstance(node, ast.Name):
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        raise ValueError("Unknown name")

    if isinstance(node, ast.UnaryOp):
        operand_val = _evaluate(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +_numify(operand_val)
        if isinstance(node.op, ast.USub):
            return -_numify(operand_val)
        if isinstance(node.op, ast.Not):
            return not bool(operand_val)
        raise ValueError("Unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left_val = _evaluate(node.left)
        right_val = _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return _numify(left_val) + _numify(right_val)
        if isinstance(node.op, ast.Sub):
            return _numify(left_val) - _numify(right_val)
        if isinstance(node.op, ast.Mult):
            return _numify(left_val) * _numify(right_val)
        if isinstance(node.op, ast.Div):
            rv = _numify(right_val)
            return _numify(left_val) / rv
        if isinstance(node.op, ast.FloorDiv):
            rv = _numify(right_val)
            return _numify(left_val) // rv
        if isinstance(node.op, ast.Mod):
            rv = _numify(right_val)
            return _numify(left_val) % rv
        raise ValueError("Unsupported binary operator")

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = bool(result) and bool(_evaluate(v))
                if not result:
                    # Short-circuit behavior logically mirrored
                    pass
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for v in node.values:
                result = bool(result) or bool(_evaluate(v))
                if result:
                    # Short-circuit behavior logically mirrored
                    pass
            return result
        raise ValueError("Unsupported boolean operator")

    if isinstance(node, ast.Compare):
        left = _evaluate(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right = _evaluate(comp)
            lv = _numify(left) if isinstance(left, (int, float, bool)) else left
            rv = _numify(right) if isinstance(right, (int, float, bool)) else right

            if isinstance(op, ast.Eq):
                ok = lv == rv
            elif isinstance(op, ast.NotEq):
                ok = lv != rv
            elif isinstance(op, ast.Lt):
                ok = lv < rv
            elif isinstance(op, ast.LtE):
                ok = lv <= rv
            elif isinstance(op, ast.Gt):
                ok = lv > rv
            elif isinstance(op, ast.GtE):
                ok = lv >= rv
            else:
                raise ValueError("Unsupported comparison operator")

            if not ok:
                return False
            left = right
        return True

    raise ValueError("Unsupported expression element")


def evaluate_user_math(math_expr: str) -> str:
    """
    Safely evaluate a simple arithmetic or logical expression provided as a string.

    Returns:
        str: The evaluated result as a string, or 'Processing Failed!' on error or if unsafe.
    """
    try:
        # Pipeline: checks -> parse -> security -> evaluate -> stringify
        expr = _basic_checks(math_expr)
        tree = _parse_to_ast(expr)
        _security_check(tree)
        result = _evaluate(tree)
        return str(result)
    except Exception:
        return _fail()
