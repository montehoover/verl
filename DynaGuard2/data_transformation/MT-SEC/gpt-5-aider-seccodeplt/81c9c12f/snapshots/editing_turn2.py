import ast
import operator
from typing import Union

Number = Union[int, float]

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


def _check_safety(node: ast.AST) -> Union[None, str]:
    """
    Returns None if the AST is safe, otherwise returns a string reason.
    """
    if isinstance(node, ast.Expression):
        return _check_safety(node.body)

    # Numbers
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return None
        return "only numeric literals are allowed"
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # for older Python versions
        return None  # type: ignore[return-value]

    # Binary operations
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BIN_OPS:
            return "only +, -, *, / operators are allowed"
        left_reason = _check_safety(node.left)
        if left_reason:
            return left_reason
        right_reason = _check_safety(node.right)
        if right_reason:
            return right_reason
        return None

    # Unary operations
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY_OPS:
            return "only unary + and - are allowed"
        return _check_safety(node.operand)

    # Disallow everything else (names, calls, attributes, etc.)
    return f"contains unsupported element: {type(node).__name__}"


def evaluate_expression(expression: str) -> Union[Number, str]:
    """
    Evaluate a simple arithmetic expression given as a string.

    Supported:
    - Addition (+), subtraction (-), multiplication (*), division (/)
    - Parentheses
    - Unary plus/minus (e.g., -3, +4)

    Returns:
        - The numeric result if the expression is safe and valid.
        - A warning string starting with 'Unsafe expression:' if the input is unsafe.
    """
    if expression is None:
        return "Unsafe expression: input must be a non-empty string."
    expr = expression.strip()
    if not expr:
        return "Unsafe expression: input must be a non-empty string."

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        return "Unsafe expression: invalid syntax."

    reason = _check_safety(parsed)
    if reason:
        return f"Unsafe expression: {reason}"

    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numbers
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type during evaluation: {type(node.value).__name__}")
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # for older Python versions
            return node.n  # type: ignore[attr-defined]

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type is ast.Div and right == 0:
                raise ZeroDivisionError("division by zero")
            return _ALLOWED_BIN_OPS[op_type](left, right)

        # Unary operations
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            return _ALLOWED_UNARY_OPS[op_type](operand)

        raise ValueError(f"Unsupported expression element during evaluation: {type(node).__name__}")

    return _eval(parsed)
