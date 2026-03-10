import re
import ast
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


def _tokenize_expression(expr: str) -> List[Union[int, str]]:
    """
    Convert the expression string into tokens: integers, operators (+,-,*,/), and parentheses.
    Handles unary + and - by folding them into numbers when applicable or by
    transforming '-( ... )' into '0 - ( ... )'.
    """
    tokens: List[Union[int, str]] = []
    n = len(expr)
    i = 0
    prev: str = "start"  # one of: start, number, op, (, )

    while i < n:
        ch = expr[i]

        if ch.isspace():
            i += 1
            continue

        if ch.isdigit():
            start = i
            while i < n and expr[i].isdigit():
                i += 1
            value = int(expr[start:i])
            tokens.append(value)
            prev = "number"
            continue

        if ch in "+-":
            is_unary = prev in ("start", "op", "(")
            # Find next non-space character
            j = i + 1
            while j < n and expr[j].isspace():
                j += 1

            if is_unary:
                if j < n and expr[j].isdigit():
                    k = j
                    while k < n and expr[k].isdigit():
                        k += 1
                    value = int(expr[j:k])
                    if ch == "-":
                        value = -value
                    tokens.append(value)
                    prev = "number"
                    i = k
                    continue
                elif j < n and expr[j] == "(":
                    # Transform '-(' or '+(' into '0 - (' or '0 + ('
                    tokens.append(0)
                    tokens.append(ch)
                    prev = "op"
                    i = j  # process '(' in the next iteration
                    continue
                else:
                    raise ValueError("Invalid use of unary operator.")
            else:
                tokens.append(ch)
                prev = "op"
                i += 1
                continue

        if ch in "*/":
            if prev not in ("number", ")"):
                raise ValueError("Operator placement is invalid.")
            tokens.append(ch)
            prev = "op"
            i += 1
            continue

        if ch == "(":
            tokens.append(ch)
            prev = "("
            i += 1
            continue

        if ch == ")":
            if prev in ("start", "op", "("):
                raise ValueError("Empty or invalid parentheses content.")
            tokens.append(ch)
            prev = ")"
            i += 1
            continue

        # Should not reach here due to prior validation
        raise ValueError(f"Unexpected character: {ch!r}")

    return tokens


def _evaluate_with_parentheses(tokens: List[Union[int, str]]) -> Number:
    """
    Evaluate tokens that may include parentheses by reducing inner expressions
    using evaluate_expression.
    """
    stack: List[List[Union[int, str]]] = [[]]

    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError("Mismatched parentheses.")
            inner = stack.pop()
            if len(inner) == 0:
                raise ValueError("Empty parentheses.")
            value = evaluate_expression(inner)
            stack[-1].append(value)
        else:
            stack[-1].append(tok)

    if len(stack) != 1:
        raise ValueError("Mismatched parentheses.")

    return evaluate_expression(stack[0])


def compute_expression(expr: str) -> Number:
    """
    Validate and compute the result of a basic arithmetic expression string.
    Supports integers, +, -, *, /, parentheses, and spaces.

    Returns:
        Number (int or float) result of the computation.

    Raises:
        ValueError: If the expression is invalid or cannot be evaluated safely.
        ZeroDivisionError: If division by zero occurs.
    """
    if not is_valid_expression(expr):
        raise ValueError("Invalid expression.")

    tokens = _tokenize_expression(expr)
    return _evaluate_with_parentheses(tokens)
