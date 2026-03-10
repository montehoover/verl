import re


_ALLOWED_EXPRESSION_RE = re.compile(r'^[0-9+\-*/().%\s^]*\Z')


def calculate_result(expr: str) -> int:
    """
    Returns the length of the input expression string if it is considered safe
    and structurally valid. Raises ValueError for unsafe characters or errors.

    Safety/validity rules:
    - Only digits, whitespace, and the operators/characters: + - * / % ^ ( ) .
    - Expression must not be empty or only whitespace.
    - Parentheses must be balanced.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")

    if expr.strip() == "":
        raise ValueError("Expression cannot be empty.")

    if not _ALLOWED_EXPRESSION_RE.match(expr):
        raise ValueError("Expression contains unsafe characters.")

    # Basic parentheses balance check
    balance = 0
    for ch in expr:
        if ch == "(":
            balance += 1
        elif ch == ")":
            balance -= 1
            if balance < 0:
                raise ValueError("Unbalanced parentheses.")
    if balance != 0:
        raise ValueError("Unbalanced parentheses.")

    return len(expr)
