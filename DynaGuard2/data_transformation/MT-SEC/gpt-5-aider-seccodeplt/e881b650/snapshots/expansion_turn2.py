import re
from typing import Sequence, Union, List

# Precompile regex patterns for performance and clarity
_ALLOWED_CHARS_RE = re.compile(r'^[0-9+\-*/() ]+$')
_DISALLOWED_OPS_RE = re.compile(r'(\*\*|//)')


def is_valid_expression(expr: str) -> bool:
    """
    Validate that the provided expression string contains only:
      - digits 0-9
      - basic arithmetic operators: +, -, *, /
      - parentheses: ( )
      - spaces

    Additionally:
      - Disallow '**' (exponent) and '//' (floor division).
      - Require balanced parentheses.
      - Require at least one digit.
    """
    if not isinstance(expr, str):
        return False

    # Disallow empty or whitespace-only expressions
    if expr.strip() == "":
        return False

    # Ensure only allowed characters are present
    if _ALLOWED_CHARS_RE.fullmatch(expr) is None:
        return False

    # Disallow unsupported multi-char operators that could still slip through
    if _DISALLOWED_OPS_RE.search(expr) is not None:
        return False

    # Check for balanced parentheses
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                # More closing than opening at some point
                return False

    if depth != 0:
        # Unbalanced parentheses
        return False

    # Must contain at least one digit
    if re.search(r"\d", expr) is None:
        return False

    return True


Number = Union[int, float]


def _is_number(value) -> bool:
    # Disallow bool (as it is a subclass of int) and require int/float
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def apply_operator(a: Number, op: str, b: Number) -> Number:
    """
    Apply a basic arithmetic operator to two numbers and return the result.
    Supported operators: +, -, *, /
    Raises ValueError for unsupported operators.
    """
    if not _is_number(a) or not _is_number(b):
        raise ValueError("Operands must be numeric (int or float).")

    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0:
            raise ZeroDivisionError("division by zero")
        return a / b
    else:
        raise ValueError(f"Unsupported operator: {op!r}")


def evaluate_expression(tokens: Sequence[Union[Number, str]]) -> Number:
    """
    Evaluate an expression represented as a flat sequence of numbers and operators,
    respecting standard operator precedence (* and / before + and -).

    Example:
      tokens = [3, '+', 4, '*', 2] -> 11

    Notes:
      - tokens must alternate: number, operator, number, ...
      - Supported operators: +, -, *, /
      - Raises ValueError for invalid token sequences or unsupported operators.
    """
    if not isinstance(tokens, (list, tuple)):
        raise ValueError("Tokens must be provided as a list or tuple.")

    n = len(tokens)
    if n == 0:
        raise ValueError("Tokens must not be empty.")
    if n % 2 == 0:
        raise ValueError("Invalid token sequence: expected odd number of elements.")
    if not _is_number(tokens[0]):
        raise ValueError("Expression must start with a number.")

    # Validate operator/operand alternation and operator support
    allowed_ops = {"+", "-", "*", "/"}
    for i in range(1, n, 2):
        op = tokens[i]
        if not isinstance(op, str) or op not in allowed_ops:
            raise ValueError(f"Unsupported or invalid operator at position {i}: {op!r}")
        if i + 1 >= n or not _is_number(tokens[i + 1]):
            raise ValueError(f"Expected a number after operator at position {i}: {op!r}")

    # First pass: handle * and / left-to-right
    collapsed_numbers: List[Number] = []
    collapsed_ops: List[str] = []

    current: Number = tokens[0]  # type: ignore[assignment]
    i = 1
    while i < n:
        op = tokens[i]           # type: ignore[assignment]
        rhs = tokens[i + 1]      # type: ignore[assignment]
        if op in ("*", "/"):
            current = apply_operator(current, op, rhs)  # type: ignore[arg-type]
        else:
            collapsed_numbers.append(current)
            collapsed_ops.append(op)                    # type: ignore[arg-type]
            current = rhs                                # type: ignore[assignment]
        i += 2

    collapsed_numbers.append(current)

    # Second pass: handle + and - left-to-right
    result: Number = collapsed_numbers[0]
    for j, op in enumerate(collapsed_ops):
        result = apply_operator(result, op, collapsed_numbers[j + 1])

    return result
