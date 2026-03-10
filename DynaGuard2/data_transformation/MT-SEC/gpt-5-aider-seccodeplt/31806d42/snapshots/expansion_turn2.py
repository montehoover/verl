import re
from typing import List, Union


def is_valid_expression(expr: str) -> bool:
    """
    Validate that the expression contains only digits, spaces, basic arithmetic operators
    (+, -, *, /), and parentheses. Also ensures parentheses are balanced and that there is
    at least one digit present. Returns True if valid, False otherwise.
    """
    if not isinstance(expr, str):
        return False

    # Must contain only allowed characters
    if not re.fullmatch(r"[0-9+\-*/()\s]+", expr):
        return False

    # Require at least one digit
    if not re.search(r"\d", expr):
        return False

    # Check balanced parentheses
    balance = 0
    for ch in expr:
        if ch == '(':
            balance += 1
        elif ch == ')':
            balance -= 1
            if balance < 0:
                return False
    if balance != 0:
        return False

    return True


NumberLike = Union[int, float, str]


def apply_operator(a: float, op: str, b: float) -> float:
    """
    Apply a basic arithmetic operator to two numbers.

    Supported operators: +, -, *, /

    Raises:
        ValueError: If an unsupported operator is provided.
        ZeroDivisionError: If division by zero is attempted.
    """
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        return a / b
    raise ValueError(f"Unsupported operator: {op!r}")


def _to_number(value: NumberLike) -> float:
    """
    Convert a token to a float. Accepts int, float, or a numeric string
    with optional leading sign and optional decimal part.

    Raises:
        ValueError: If the token cannot be interpreted as a number.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return float(value)
    raise ValueError(f"Invalid numeric token: {value!r}")


def compute_with_precedence(tokens: List[NumberLike]) -> float:
    """
    Compute the result of an expression represented as a list of alternating
    numbers and operators, respecting standard operator precedence (* and /
    before + and -). Evaluation is left-to-right within the same precedence.

    Example:
        tokens = [3, '+', 4, '*', 2] -> 11.0

    Args:
        tokens: A list like [number, op, number, op, number, ...]
                Numbers may be int, float, or numeric strings.
                Operators must be one of '+', '-', '*', '/'.

    Returns:
        The computed result as a float.

    Raises:
        ValueError: If tokens are malformed or contain unsupported operators.
        ZeroDivisionError: If division by zero occurs.
    """
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("Tokens must be a non-empty list.")

    if len(tokens) % 2 == 0:
        raise ValueError("Tokens must alternate number and operator, starting and ending with a number.")

    # Split into values and operators ensuring the correct alternating pattern
    values: List[float] = []
    ops: List[str] = []

    for i, tok in enumerate(tokens):
        if i % 2 == 0:
            # Expect a number
            values.append(_to_number(tok))
        else:
            # Expect an operator
            if not isinstance(tok, str):
                raise ValueError(f"Operator token must be a string, got {type(tok).__name__}")
            if tok not in {"+", "-", "*", "/"}:
                # Raise ValueError for unsupported operator as requested
                raise ValueError(f"Unsupported operator: {tok!r}")
            ops.append(tok)

    # First pass: handle * and / with left-to-right evaluation
    reduced_values: List[float] = [values[0]]
    reduced_ops: List[str] = []

    for i, op in enumerate(ops):
        if op in ("*", "/"):
            reduced_values[-1] = apply_operator(reduced_values[-1], op, values[i + 1])
        else:
            reduced_ops.append(op)
            reduced_values.append(values[i + 1])

    # Second pass: handle + and - left-to-right
    result = reduced_values[0]
    for i, op in enumerate(reduced_ops):
        result = apply_operator(result, op, reduced_values[i + 1])

    return result
