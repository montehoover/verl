import re
from typing import Pattern, List, Union

Number = Union[int, float]
Token = Union[Number, str]

_ALLOWED_CHARS_PATTERN: Pattern[str] = re.compile(r'^[0-9+\-*/() ]+$')

def validate_expression(expr: str) -> bool:
    """
    Return True if expr contains only digits, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces; otherwise return False.
    Uses regex for validation.
    """
    if not isinstance(expr, str):
        return False
    return _ALLOWED_CHARS_PATTERN.fullmatch(expr) is not None


def apply_operator(a: Number, b: Number, op: str) -> Number:
    """
    Apply a basic arithmetic operator to two numbers and return the result.
    Supported operators: +, -, *, /
    Raises ValueError for unsupported operators.
    """
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        return a / b
    raise ValueError(f"Unsupported operator: {op}")


def calculate_with_precedence(tokens: List[Token]) -> Number:
    """
    Evaluate a flat list of numbers and operators with correct precedence.
    Example tokens: [3, '+', 4, '*', 5] -> 23
    Supports operators: +, -, *, /
    Raises ValueError for invalid token sequences or unsupported operators.
    """
    if not tokens:
        raise ValueError("Empty token list")

    if len(tokens) % 2 == 0:
        raise ValueError("Invalid expression: tokens must alternate and start/end with a number")

    # Validate first token
    if not isinstance(tokens[0], (int, float)):
        raise ValueError("Invalid token at position 0: expected a number")

    # First pass: handle * and /, compressing them
    first_pass: List[Token] = []
    current: Number = tokens[0]  # guaranteed number from the check above
    i = 1
    while i < len(tokens):
        op = tokens[i]
        if not isinstance(op, str):
            raise ValueError(f"Invalid token at position {i}: expected an operator")

        if i + 1 >= len(tokens):
            raise ValueError("Operator at end of expression")

        nxt = tokens[i + 1]
        if not isinstance(nxt, (int, float)):
            raise ValueError(f"Invalid token at position {i + 1}: expected a number")

        if op in ('*', '/'):
            current = apply_operator(current, nxt, op)
        elif op in ('+', '-'):
            first_pass.append(current)
            first_pass.append(op)
            current = nxt
        else:
            raise ValueError(f"Unsupported operator: {op}")

        i += 2

    first_pass.append(current)

    # Second pass: handle + and -
    result: Number = first_pass[0]  # should be a number
    j = 1
    while j < len(first_pass):
        op = first_pass[j]
        nxt = first_pass[j + 1]
        result = apply_operator(result, nxt, op)  # op should be '+' or '-'
        j += 2

    return result
