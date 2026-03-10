import ast
import operator
from typing import Any

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_expr(node: ast.AST) -> Any:
    """
    Evaluate a safe subset of Python expressions:
    - numbers (int/float) and strings
    - parentheses
    - operators: +, -, *, /
      * '+' supports numeric addition and string concatenation
      * '*' supports numeric multiplication and string repetition by an integer
    """
    def _eval(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp):
            op_type = type(n.op)
            if op_type not in _ALLOWED_BIN_OPS:
                raise ValueError("Disallowed operator")
            left = _eval(n.left)
            right = _eval(n.right)
            if op_type is ast.Div and right == 0:
                raise ZeroDivisionError("division by zero")
            return _ALLOWED_BIN_OPS[op_type](left, right)

        if isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            if op_type not in _ALLOWED_UNARY_OPS:
                raise ValueError("Disallowed unary operator")
            operand = _eval(n.operand)
            return _ALLOWED_UNARY_OPS[op_type](operand)

        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, bool):
                raise ValueError("Booleans not allowed")
            if isinstance(val, (int, float, str)):
                return val
            raise ValueError("Unsupported constant type")

        # For Python <3.8 compatibility
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            val = n.n
            if isinstance(val, bool):
                raise ValueError("Booleans not allowed")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Unsupported numeric constant")
        if hasattr(ast, "Str") and isinstance(n, ast.Str):  # type: ignore[attr-defined]
            return n.s

        # Disallow names, calls, attributes, subscripts, comprehensions, f-strings, etc.
        raise ValueError("Disallowed expression node")

    return _eval(node)


def calculate_expression(expression: str) -> Any:
    """
    Safely evaluate a simple arithmetic or string expression consisting of:
    - numbers (int/float) and strings
    - parentheses
    - operators: +, -, *, /
      * '+' supports numeric addition and string concatenation
      * '*' supports numeric multiplication and string repetition by an integer

    Returns 'Execution Blocked!' for invalid or unsafe input.
    """
    if not isinstance(expression, str):
        return "Execution Blocked!"

    try:
        parsed = ast.parse(expression, mode="eval")
    except Exception:
        return "Execution Blocked!"

    try:
        return _safe_eval_expr(parsed)
    except Exception:
        return "Execution Blocked!"


def execute_user_code(script_code: str) -> Any:
    """
    Safely execute a user-supplied Python script string that may only contain a
    single expression using basic arithmetic and string operations.
    Returns the value of that expression, or 'Execution Blocked!' if unsafe.
    """
    if not isinstance(script_code, str):
        return "Execution Blocked!"

    try:
        module = ast.parse(script_code, mode="exec")
    except Exception:
        return "Execution Blocked!"

    # Allow only a single expression statement (no assignments, imports, etc.)
    if not isinstance(module, ast.Module) or len(module.body) != 1:
        return "Execution Blocked!"
    stmt = module.body[0]
    if not isinstance(stmt, ast.Expr):
        return "Execution Blocked!"

    try:
        return _safe_eval_expr(stmt.value)
    except Exception:
        return "Execution Blocked!"
