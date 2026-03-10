import re
import ast

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


def compute_expression(input_expr: str) -> float:
    """
    Validate and compute the result of a simple arithmetic expression string.
    Supports digits, spaces, +, -, *, /, and parentheses. Respects operator precedence.
    Raises ValueError for invalid input or malformed expressions.
    """
    if not is_valid_expression(input_expr):
        raise ValueError("Invalid expression")

    expr = input_expr.replace(' ', '')

    # Tokenize expression into numbers, operators, and parentheses.
    tokens = []
    i = 0
    prev_type = 'start'  # one of: start, num, op, (, )
    while i < len(expr):
        ch = expr[i]

        if ch.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            num = int(expr[i:j])
            tokens.append(num)
            prev_type = 'num'
            i = j
            continue

        if ch in '+-':
            # Determine unary vs binary
            if prev_type in ('start', 'op', '('):
                # Unary sign
                if i + 1 < len(expr) and expr[i + 1].isdigit():
                    j = i + 1
                    while j < len(expr) and expr[j].isdigit():
                        j += 1
                    num = int(expr[i + 1:j])
                    if ch == '-':
                        num = -num
                    tokens.append(num)
                    prev_type = 'num'
                    i = j
                    continue
                elif i + 1 < len(expr) and expr[i + 1] == '(':
                    # Convert unary sign before '(' into 0 <op> (...)
                    tokens.append(0)
                    tokens.append(ch)
                    prev_type = 'op'
                    i += 1
                    continue
                else:
                    raise ValueError("Invalid expression")
            # Binary operator
            tokens.append(ch)
            prev_type = 'op'
            i += 1
            continue

        if ch in '*/':
            tokens.append(ch)
            prev_type = 'op'
            i += 1
            continue

        if ch == '(':
            tokens.append(ch)
            prev_type = '('
            i += 1
            continue

        if ch == ')':
            tokens.append(ch)
            prev_type = ')'
            i += 1
            continue

        # Should not occur due to validation regex
        raise ValueError("Invalid expression")

    # Resolve parentheses using a stack, evaluating inner expressions first.
    stack = []
    for t in tokens:
        if t != ')':
            stack.append(t)
        else:
            sub = []
            while stack and stack[-1] != '(':
                sub.append(stack.pop())
            if not stack or stack[-1] != '(':
                raise ValueError("Mismatched parentheses")
            stack.pop()  # remove '('
            sub.reverse()
            value = evaluate_expression(sub)
            stack.append(value)

    # Ensure no unmatched '(' remain
    if any(tok == '(' for tok in stack):
        raise ValueError("Mismatched parentheses")

    # Evaluate the remaining (parentheses-free) expression
    return evaluate_expression(stack)
