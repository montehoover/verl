import re

_ALLOWED_CHARS_PATTERN = re.compile(r'^[0-9+\-*/ ]*$')

def validate_math_expression(value):
    """
    Return True if value contains only digits (0-9), '+', '-', '*', '/', and spaces.
    Otherwise return False.
    """
    if not isinstance(value, str):
        return False
    return _ALLOWED_CHARS_PATTERN.fullmatch(value) is not None


def compute(a, op, b):
    """
    Compute a single binary arithmetic operation.

    Parameters:
        a (float or int): Left operand.
        op (str): One of '+', '-', '*', '/'.
        b (float or int): Right operand.

    Returns:
        float: The result of the operation.

    Raises:
        ValueError: If an unsupported operator is provided.
        ZeroDivisionError: If division by zero is attempted.
    """
    a = float(a)
    b = float(b)
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        if b == 0.0:
            raise ZeroDivisionError("division by zero")
        return a / b
    raise ValueError(f"Unsupported operator: {op}")


def parse_expression(expr):
    """
    Parse and evaluate a mathematical expression string containing only digits,
    '+', '-', '*', '/', and spaces. Supports unary plus/minus and respects
    standard operator precedence (* and / before + and -). Does not use eval.

    Parameters:
        expr (str): The expression to parse.

    Returns:
        float: The computed result.

    Raises:
        TypeError: If expr is not a string.
        ValueError: If the expression contains invalid characters or syntax.
        ZeroDivisionError: If division by zero occurs.
    """
    if not isinstance(expr, str):
        raise TypeError("Expression must be a string")

    if not validate_math_expression(expr):
        raise ValueError("Expression contains invalid characters")

    tokens = _tokenize(expr)
    return _evaluate_tokens(tokens)


def _tokenize(expr):
    """
    Convert the input expression string into a list of tokens (numbers and operators).
    Merges unary '+'/'-' into signed numbers.
    """
    tokens = []
    i = 0
    n = len(expr)
    prev_kind = 'start'  # 'start' | 'num' | 'op'

    while i < n:
        ch = expr[i]

        if ch.isspace():
            i += 1
            continue

        if ch in '+-':
            if prev_kind in ('start', 'op'):
                # Parse a run of unary +/-
                sign = 1
                while i < n and expr[i] in '+-':
                    if expr[i] == '-':
                        sign *= -1
                    i += 1
                # Next must be digits
                start = i
                while i < n and expr[i].isdigit():
                    i += 1
                if i == start:
                    raise ValueError("Expected number after unary sign(s)")
                num = int(expr[start:i]) * sign
                tokens.append(float(num))
                prev_kind = 'num'
                continue
            else:
                # Binary + or -
                tokens.append(ch)
                prev_kind = 'op'
                i += 1
                continue

        if ch in '*/':
            if prev_kind in ('start', 'op'):
                # '*' or '/' cannot be unary
                raise ValueError("Unexpected operator position")
            tokens.append(ch)
            prev_kind = 'op'
            i += 1
            continue

        if ch.isdigit():
            start = i
            while i < n and expr[i].isdigit():
                i += 1
            num = int(expr[start:i])
            tokens.append(float(num))
            prev_kind = 'num'
            continue

        # Should never reach here due to validation, but just in case:
        raise ValueError(f"Invalid character encountered: {ch!r}")

    if not tokens:
        raise ValueError("Empty expression")

    if prev_kind != 'num':
        raise ValueError("Expression cannot end with an operator")

    # Sanity check: tokens should follow num (op num)* pattern
    expect = 'num'
    for t in tokens:
        if expect == 'num':
            if isinstance(t, float):
                expect = 'op'
            else:
                raise ValueError("Invalid token order: expected number")
        else:  # expect operator
            if isinstance(t, str):
                expect = 'num'
            else:
                raise ValueError("Invalid token order: expected operator")

    return tokens


def _evaluate_tokens(tokens):
    """
    Evaluate a token list using the shunting-yard algorithm to respect operator precedence.
    """
    values = []
    ops = []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

    def reduce_once():
        if len(values) < 2 or not ops:
            raise ValueError("Invalid expression")
        b = values.pop()
        a = values.pop()
        op = ops.pop()
        values.append(compute(a, op, b))

    for t in tokens:
        if isinstance(t, float):
            values.append(t)
        else:
            while ops and precedence[ops[-1]] >= precedence[t]:
                reduce_once()
            ops.append(t)

    while ops:
        reduce_once()

    if len(values) != 1:
        raise ValueError("Invalid expression")

    return values[0]
