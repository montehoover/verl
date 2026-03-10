import re

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
