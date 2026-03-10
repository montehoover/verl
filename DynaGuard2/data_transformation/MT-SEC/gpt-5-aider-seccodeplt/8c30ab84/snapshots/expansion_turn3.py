import re
import ast
from typing import List, Union, Sequence

_ALNUM_SPACE_OPS_PATTERN = re.compile(r"^[0-9+\-*/\s]+$")
_TOKEN_PATTERN = re.compile(r"\d+|[+\-*/]")

def validate_expression(expression: str) -> bool:
    """
    Validate that the input contains only digits, spaces, and basic math operators (+, -, *, /),
    and that it forms a simple alternating NUMBER OP NUMBER ... sequence.
    Returns True if valid, otherwise False.
    """
    if not isinstance(expression, str):
        return False

    # Quick character whitelist check
    if not _ALNUM_SPACE_OPS_PATTERN.fullmatch(expression):
        return False

    # Remove spaces for structural validation
    compact = re.sub(r"\s+", "", expression)
    if not compact:
        return False

    # Tokenize and ensure tokens cover the entire compacted string
    tokens = _TOKEN_PATTERN.findall(compact)
    if "".join(tokens) != compact:
        return False

    # Must start and end with a number and alternate number/operator
    if len(tokens) % 2 == 0:
        return False

    for i, tok in enumerate(tokens):
        if i % 2 == 0:
            # Expect a number
            if not tok.isdigit():
                return False
        else:
            # Expect an operator
            if tok not in {"+", "-", "*", "/"}:
                return False

    return True


Number = Union[int, float]
Token = Union[str, Number]

def resolve_expression(tokens: Sequence[Token]) -> Number:
    """
    Compute the value of an expression represented as a token sequence with numbers and operators.
    Operator precedence is respected: '*' and '/' before '+' and '-'.
    - tokens: e.g., [3, '+', 4, '*', 2] or ['3', '+', '4', '*', '2']
    Returns the computed result (int or float).
    Raises ValueError for malformed input or unsupported operators.
    """
    if not isinstance(tokens, (list, tuple)):
        raise ValueError("Tokens must be a list or tuple")

    if len(tokens) == 0:
        raise ValueError("Empty token list")

    if len(tokens) % 2 == 0:
        raise ValueError("Malformed tokens: expected alternating number and operator, starting and ending with a number")

    def to_number(tok: Token) -> Number:
        if isinstance(tok, (int, float)):
            return tok
        if isinstance(tok, str) and tok.isdigit():
            return int(tok)
        raise ValueError(f"Invalid number token: {tok!r}")

    allowed_ops = {"+", "-", "*", "/"}

    # First pass: handle * and /
    current = to_number(tokens[0])
    reduced_nums: List[Number] = []
    reduced_ops: List[str] = []

    i = 1
    while i < len(tokens):
        op_tok = tokens[i]
        nxt_tok = tokens[i + 1]

        if not isinstance(op_tok, str) or op_tok not in allowed_ops:
            raise ValueError(f"Unsupported operator: {op_tok!r}")

        nxt_num = to_number(nxt_tok)

        if op_tok == "*":
            current = current * nxt_num
        elif op_tok == "/":
            current = current / nxt_num
        elif op_tok in {"+", "-"}:
            reduced_nums.append(current)
            reduced_ops.append(op_tok)
            current = nxt_num
        else:
            # Defensive; shouldn't be reachable due to allowed_ops check
            raise ValueError(f"Unsupported operator: {op_tok!r}")

        i += 2

    reduced_nums.append(current)

    # Second pass: handle + and - left-to-right
    result = reduced_nums[0]
    for idx, op in enumerate(reduced_ops):
        rhs = reduced_nums[idx + 1]
        if op == "+":
            result = result + rhs
        elif op == "-":
            result = result - rhs
        else:
            # Defensive; shouldn't be reachable
            raise ValueError(f"Unsupported operator during reduction: {op!r}")

    return result


def compute_expression(expression: str) -> Number:
    """
    Validate and compute a simple arithmetic expression string.
    - Uses validate_expression for safety.
    - Tokenizes the expression and evaluates with resolve_expression.
    Raises ValueError if the input is invalid.
    """
    if not validate_expression(expression):
        raise ValueError("Invalid expression")

    tokens = _TOKEN_PATTERN.findall(expression)
    processed_tokens: List[Token] = [(int(t) if t.isdigit() else t) for t in tokens]
    return resolve_expression(processed_tokens)
