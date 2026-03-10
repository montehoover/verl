import re

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
