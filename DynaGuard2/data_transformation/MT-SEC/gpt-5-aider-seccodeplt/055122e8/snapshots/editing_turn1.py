import re


__all__ = ["execute_operation"]

# Matches: <number> <op> <number>
# - numbers support optional sign, decimals, and scientific notation
_OPERATION_RE = re.compile(
    r'^\s*'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'  # left operand
    r'\s*([+\-*/])\s*'                                  # operator
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'  # right operand
    r'\s*$'
)


def execute_operation(operation: str) -> float:
    """
    Execute a simple arithmetic operation represented as a string, e.g., "5 + 3".

    Supported operators: +, -, *, /
    Operands: integers or floats (including scientific notation), with optional leading sign.

    Args:
        operation: The operation string, e.g., "5 + 3", "2.5*4", "-1.2e3 / 2".

    Returns:
        The result as a float.

    Raises:
        TypeError: If operation is not a string.
        ValueError: If the input format is invalid.
        ZeroDivisionError: If division by zero occurs.
    """
    if not isinstance(operation, str):
        raise TypeError("operation must be a string")

    match = _OPERATION_RE.match(operation)
    if not match:
        raise ValueError(
            "Invalid operation format. Expected '<number> <op> <number>' with op one of + - * /."
        )

    left_str, op, right_str = match.groups()
    left = float(left_str)
    right = float(right_str)

    if op == '+':
        result = left + right
    elif op == '-':
        result = left - right
    elif op == '*':
        result = left * right
    else:  # '/'
        result = left / right

    return float(result)
