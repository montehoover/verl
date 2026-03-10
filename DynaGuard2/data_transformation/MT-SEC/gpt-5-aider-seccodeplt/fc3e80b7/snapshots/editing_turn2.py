import ast
import operator
from typing import Union

_BLOCKED = "Execution Blocked!"

# Operators allowed for evaluation
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _is_number(value) -> bool:
    # Disallow booleans (bool is a subclass of int)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_string(value) -> bool:
    return isinstance(value, str)


def _eval_node(node: ast.AST) -> Union[int, float, str]:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError("disallowed operator")
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        # Addition supports both numeric addition and string concatenation
        if op_type is ast.Add:
            if _is_number(left) and _is_number(right):
                return operator.add(left, right)
            if _is_string(left) and _is_string(right):
                return left + right
            raise ValueError("type mismatch for +")

        # Other binary ops require numeric operands only
        if not (_is_number(left) and _is_number(right)):
            raise ValueError("non-numeric operand")
        return _BINARY_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError("disallowed unary operator")
        operand = _eval_node(node.operand)
        if not _is_number(operand):
            raise ValueError("non-numeric operand")
        return _UNARY_OPS[op_type](operand)

    # Numbers and strings: support both Constant (py3.8+) and Num/Str (older)
    if isinstance(node, ast.Constant):
        if _is_number(node.value) or _is_string(node.value):
            return node.value
        raise ValueError("disallowed constant")
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        if _is_number(node.n):  # type: ignore[attr-defined]
            return node.n  # type: ignore[attr-defined]
        raise ValueError("disallowed num")
    if hasattr(ast, "Str") and isinstance(node, ast.Str):  # type: ignore[attr-defined]
        if _is_string(node.s):  # type: ignore[attr-defined]
            return node.s  # type: ignore[attr-defined]
        raise ValueError("disallowed str")

    # Everything else is blocked (Names, Calls, Attributes, etc.)
    raise ValueError("disallowed syntax")


def evaluate_expression(expression: str) -> Union[int, float, str]:
    """
    Safely evaluate a simple expression given as a string.
    Supports:
      - Numeric operations: +, -, *, /, unary +/-, and parentheses
      - String concatenation using +

    Returns:
      - numeric or string result on success
      - 'Execution Blocked!' if the expression is invalid or unsafe
    """
    if not isinstance(expression, str):
        return _BLOCKED

    try:
        # Parse expression into an AST in eval mode
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        # Result must be a number or string
        if not (_is_number(result) or _is_string(result)):
            return _BLOCKED
        return result
    except Exception:
        return _BLOCKED


if __name__ == "__main__":
    # Simple manual test runner when executed directly
    samples = [
        "1 + 2 * 3",
        "(1 + 2) * 3",
        "-4 + +5",
        "10 / 2",
        "10 / 0",
        "2 ** 3",
        "__import__('os').system('echo hi')",
        "True + 1",
        "3.5 - 1.2 * 2",
        "'hello' + ' world'",
        "\"foo\" + \"bar\"",
        "'foo' + 3",
        "3 + '4'",
    ]
    for s in samples:
        print(s, "=>", evaluate_expression(s))
