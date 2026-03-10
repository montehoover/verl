import re
from typing import Union, List

# Precompiled regex to allow only digits, basic arithmetic operators, and spaces
_ALLOWED_PATTERN = re.compile(r'^[0-9+\-*/ ]*$')

def sanitize_input(value: Union[str, bytes]) -> bool:
    """
    Validate that the input consists only of digits (0-9), +, -, *, /, and spaces.

    Args:
        value: The input to validate. Must be a string. Bytes are rejected to avoid
               implicit decoding ambiguities.

    Returns:
        True if the input is a string and contains only the allowed characters,
        otherwise False.
    """
    if not isinstance(value, str):
        return False
    return _ALLOWED_PATTERN.fullmatch(value) is not None


def parse_expression(expr: str) -> List[Union[int, str]]:
    """
    Parse a sanitized arithmetic expression string into Reverse Polish Notation (RPN),
    respecting operator precedence and associativity. Supports +, -, *, /, spaces,
    and unary minus.

    The returned list contains integers and operator tokens:
      '+', '-', '*', '/', and 'u-' for unary minus.

    Args:
        expr: A string that has already been sanitized to contain only digits,
              '+', '-', '*', '/', and spaces.

    Returns:
        A list representing the expression in RPN suitable for safe evaluation.

    Raises:
        ValueError: If the input is not a string, not sanitized, or the expression
                    is syntactically invalid.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")
    if not sanitize_input(expr):
        raise ValueError("Expression contains invalid characters.")

    # Operator precedence and associativity
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, 'u-': 3}
    assoc_left = {'+': True, '-': True, '*': True, '/': True, 'u-': False}

    output: List[Union[int, str]] = []
    op_stack: List[str] = []

    i = 0
    n = len(expr)
    prev_token_type = None  # None | 'number' | 'operator'

    def pop_while(op: str):
        while op_stack:
            top = op_stack[-1]
            if (assoc_left[op] and precedence[op] <= precedence[top]) or (
                not assoc_left[op] and precedence[op] < precedence[top]
            ):
                output.append(op_stack.pop())
            else:
                break

    while i < n:
        # Skip spaces
        if expr[i].isspace():
            i += 1
            continue

        ch = expr[i]

        # Number token
        if ch.isdigit():
            if prev_token_type == 'number':
                raise ValueError("Invalid expression: missing operator between numbers.")
            j = i + 1
            while j < n and expr[j].isdigit():
                j += 1
            value = int(expr[i:j])
            output.append(value)
            prev_token_type = 'number'
            i = j
            continue

        # Operator token
        if ch in '+-*/':
            # Determine unary minus
            if ch == '-' and (prev_token_type is None or prev_token_type == 'operator'):
                op = 'u-'
            else:
                op = ch

            # Binary operators cannot start or follow another operator
            if op != 'u-' and (prev_token_type is None or prev_token_type == 'operator'):
                raise ValueError("Invalid expression: operator in unexpected position.")

            pop_while(op)
            op_stack.append(op)
            prev_token_type = 'operator'
            i += 1
            continue

        # Should not reach here if sanitized_input is used, but guard anyway
        raise ValueError(f"Invalid character encountered: {ch!r}")

    if prev_token_type != 'number':
        raise ValueError("Invalid expression: cannot end with an operator.")

    # Drain operator stack
    while op_stack:
        output.append(op_stack.pop())

    return output
