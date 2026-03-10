import re

def is_valid_expression(expr: str) -> bool:
    if not isinstance(expr, str) or not expr:
        return False
    # Allow only digits, +, -, *, /, parentheses, and spaces
    if not re.fullmatch(r'[0-9+\-*/() ]+', expr):
        return False
    # Ensure there's at least one digit
    if not re.search(r'\d', expr):
        return False
    return True


def apply_operator(a: float, b: float, op: str) -> float:
    """
    Apply a basic arithmetic operator to two numbers.
    Supported operators: +, -, *, /
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


def evaluate_expression(tokens: list) -> float:
    """
    Evaluate an expression represented as a list of numbers and operators,
    respecting operator precedence (* and / before + and -).

    Example:
        [2, '+', 3, '*', 4] -> 14
        [10, '/', 2, '+', 3] -> 8

    The list must alternate between numbers and operators, starting and ending with a number.
    Raises ValueError for invalid structure or unsupported operators.
    """
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("Tokens must be a non-empty list")

    if len(tokens) % 2 == 0:
        raise ValueError("Invalid token sequence: must alternate number, operator, number, ...")

    # Validate tokens and supported operators
    for i, t in enumerate(tokens):
        if i % 2 == 0:
            if not isinstance(t, (int, float)):
                raise ValueError(f"Expected number at position {i}, got {type(t).__name__}")
        else:
            if not isinstance(t, str):
                raise ValueError(f"Expected operator string at position {i}, got {type(t).__name__}")
            if t not in {"+", "-", "*", "/"}:
                raise ValueError(f"Unsupported operator: {t}")

    # First pass: resolve * and / to handle precedence
    reduced = [float(tokens[0])]
    i = 1
    while i < len(tokens):
        op = tokens[i]
        num = float(tokens[i + 1])
        if op in ('*', '/'):
            left = reduced.pop()
            reduced.append(apply_operator(left, num, op))
        else:
            reduced.append(op)
            reduced.append(num)
        i += 2

    # Second pass: resolve + and -
    result = reduced[0]
    i = 1
    while i < len(reduced):
        op = reduced[i]
        num = reduced[i + 1]
        result = apply_operator(result, num, op)
        i += 2

    return result
