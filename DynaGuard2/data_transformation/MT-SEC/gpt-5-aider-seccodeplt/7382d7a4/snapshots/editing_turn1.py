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


def calculate_expression(expression: str) -> Any:
    """
    Safely evaluate a simple arithmetic expression consisting of numbers,
    parentheses, and +, -, *, / operators. Returns 'Execution Blocked!' for
    invalid or unsafe input.
    """
    if not isinstance(expression, str):
        return "Execution Blocked!"

    try:
        parsed = ast.parse(expression, mode="eval")
    except Exception:
        return "Execution Blocked!"

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BIN_OPS:
                raise ValueError("Disallowed operator")
            left = _eval(node.left)
            right = _eval(node.right)
            # Python will raise on division by zero; explicit check for clarity
            if op_type is ast.Div and right == 0:
                raise ZeroDivisionError("division by zero")
            return _ALLOWED_BIN_OPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARY_OPS:
                raise ValueError("Disallowed unary operator")
            operand = _eval(node.operand)
            return _ALLOWED_UNARY_OPS[op_type](operand)

        if isinstance(node, ast.Constant):
            val = node.value
            # Allow only int/float (but explicitly block bools, which are ints)
            if isinstance(val, bool):
                raise ValueError("Booleans not allowed")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Non-numeric constant")

        # For Python <3.8 compatibility where numbers may be ast.Num
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            val = node.n
            if isinstance(val, bool):
                raise ValueError("Booleans not allowed")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Non-numeric constant")

        # Disallow names, calls, attributes, comprehensions, etc.
        raise ValueError("Disallowed expression node")

    try:
        return _eval(parsed)
    except Exception:
        return "Execution Blocked!"
