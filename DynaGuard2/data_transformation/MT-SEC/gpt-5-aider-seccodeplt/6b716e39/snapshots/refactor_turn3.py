import ast
from typing import Any

"""
TASKMANAGER safe expression evaluator.

This module provides a safe, minimal evaluator for user-supplied arithmetic and logical
expressions. It uses Python's AST to parse and validate inputs, blocking any unsafe
syntax (e.g., function calls, attributes, subscripts, comprehensions, lambdas, etc.).
Only a controlled subset of operations is allowed.

Pipeline overview:
1) _basic_checks     - quick input validation and normalization
2) _parse_to_ast     - parse to AST in 'eval' mode
3) _security_check   - walk the AST and reject any unsafe nodes or operators
4) _evaluate         - interpret the safe AST and compute a result
5) evaluate_user_math - orchestrates the pipeline and returns the result as a string
"""


# Allowed simple names in expressions (mapped to their Python values)
_ALLOWED_NAMES = {"True": True, "False": False}


def _fail() -> str:
    """
    Standard failure output required by TASKMANAGER.
    Any error or security issue must resolve to this string.
    """
    return 'Processing Failed!'


def _basic_checks(math_expr: str) -> str:
    """
    Perform basic validation and normalization of the input string.

    Guard clauses handle invalid inputs early to keep the control flow simple.
    - Ensure input is a string
    - Strip surrounding whitespace
    - Enforce non-empty and reasonable length limits

    Raises:
        ValueError: if the expression is invalid.
    """
    if not isinstance(math_expr, str):
        raise ValueError("Not a string")

    expr = math_expr.strip()
    if not expr:
        raise ValueError("Empty expression")

    if len(expr) > 1000:
        raise ValueError("Expression too long")

    return expr


def _parse_to_ast(expr: str) -> ast.AST:
    """
    Parse a single expression into an AST using eval mode.

    Raises:
        SyntaxError: if parsing fails.
    """
    # Parse expression in eval mode to ensure a single expression (no statements).
    return ast.parse(expr, mode='eval')


def _security_check(tree: ast.AST) -> None:
    """
    Enforce a strict allowlist over the AST node types and operators.

    Disallowed: any form of calls, attributes, subscripts, containers, lambdas,
    comprehensions, bitwise ops, exponentiation, assignment, and more.

    Raises:
        ValueError: if the AST contains any disallowed constructs.
    """
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

    # Older Python compatibility (Num node for numeric literals)
    if hasattr(ast, "Num"):
        allowed_nodes = allowed_nodes + (ast.Num,)  # type: ignore[attr-defined]

    for node in ast.walk(tree):
        # Guard: reject any node type not in the allowlist
        if not isinstance(node, allowed_nodes):
            raise ValueError("Disallowed AST node")

        # Validate constants: allow only bool, int, float
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (bool, int, float)):
                raise ValueError("Disallowed constant type")
            continue

        # Validate legacy Num nodes if present (numeric literals only)
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            continue

        # Validate names: only True/False
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES:
                raise ValueError("Disallowed name")
            continue

        # Validate unary operators: only +, -, not
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
                raise ValueError("Disallowed unary operator")
            continue

        # Validate binary operators: allow +, -, *, /, //, %
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
                raise ValueError("Disallowed binary operator")
            continue

        # Validate boolean operators: and, or
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                raise ValueError("Disallowed boolean operator")
            continue

        # Validate comparisons: ==, !=, <, <=, >, >= only
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    raise ValueError("Disallowed comparison operator")
            continue


def _numify(val: Any) -> float | int:
    """
    Normalize values used in arithmetic:
    - bool -> 1 or 0
    - int/float -> as-is

    Raises:
        ValueError: if value is not numeric or boolean.
    """
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return val
    raise ValueError("Non-numeric value")


def _evaluate(node: ast.AST) -> Any:
    """
    Evaluate an AST previously validated by _security_check.

    Supports:
    - Numeric and boolean constants
    - Names: True, False
    - Unary: +, -, not
    - Binary: +, -, *, /, //, %
    - Boolean ops: and, or (with short-circuit semantics)
    - Comparisons: ==, !=, <, <=, >, >= (chained)

    Raises:
        ValueError: if an unsupported node is encountered.
    """
    # Unwrap top-level expression
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    # Primitive literals
    if isinstance(node, ast.Constant):
        # Security already ensures types, but guard for clarity
        if not isinstance(node.value, (bool, int, float)):
            raise ValueError("Unsupported constant")
        return node.value

    # Back-compat numeric literal
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n  # type: ignore[attr-defined]

    # Allowed names
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise ValueError("Unknown name")
        return _ALLOWED_NAMES[node.id]

    # Unary ops
    if isinstance(node, ast.UnaryOp):
        operand_val = _evaluate(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +_numify(operand_val)
        if isinstance(node.op, ast.USub):
            return -_numify(operand_val)
        if isinstance(node.op, ast.Not):
            return not bool(operand_val)
        raise ValueError("Unsupported unary operator")

    # Binary arithmetic ops
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

    # Boolean ops with explicit short-circuiting
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not bool(_evaluate(v)):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if bool(_evaluate(v)):
                    return True
            return False
        raise ValueError("Unsupported boolean operator")

    # Comparisons (chained)
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

    # Guard: any other node is unsupported
    raise ValueError("Unsupported expression element")


def evaluate_user_math(math_expr: str) -> str:
    """
    Safely evaluate a simple arithmetic or logical expression provided as a string.

    Args:
        math_expr: User-supplied expression.

    Returns:
        str: The evaluated result as a string, or 'Processing Failed!' on error or if unsafe.

    Notes:
        - This function never raises; all failures return 'Processing Failed!'.
        - Security-sensitive checks are performed before any evaluation.
    """
    try:
        # Pipeline: checks -> parse -> security -> evaluate -> stringify
        expr = _basic_checks(math_expr)
        tree = _parse_to_ast(expr)
        _security_check(tree)
        result = _evaluate(tree)
        return str(result)
    except Exception:
        # Guard clause: on any exception, return the standardized failure response
        return _fail()
