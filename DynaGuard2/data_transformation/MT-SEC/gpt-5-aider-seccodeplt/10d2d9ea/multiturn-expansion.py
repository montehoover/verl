import re
import ast
from typing import Pattern, List, Union

Number = Union[int, float]
Token = Union[Number, str]

_ALLOWED_CHARS_PATTERN: Pattern[str] = re.compile(r'^[0-9+\-*/() ]+$')

def validate_expression(expr: str) -> bool:
    """
    Return True if expr contains only digits, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces; otherwise return False.
    Uses regex for validation.
    """
    if not isinstance(expr, str):
        return False
    return _ALLOWED_CHARS_PATTERN.fullmatch(expr) is not None


def apply_operator(a: Number, b: Number, op: str) -> Number:
    """
    Apply a basic arithmetic operator to two numbers and return the result.
    Supported operators: +, -, *, /
    Raises ValueError for unsupported operators.
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


def calculate_with_precedence(tokens: List[Token]) -> Number:
    """
    Evaluate a flat list of numbers and operators with correct precedence.
    Example tokens: [3, '+', 4, '*', 5] -> 23
    Supports operators: +, -, *, /
    Raises ValueError for invalid token sequences or unsupported operators.
    """
    if not tokens:
        raise ValueError("Empty token list")

    if len(tokens) % 2 == 0:
        raise ValueError("Invalid expression: tokens must alternate and start/end with a number")

    # Validate first token
    if not isinstance(tokens[0], (int, float)):
        raise ValueError("Invalid token at position 0: expected a number")

    # First pass: handle * and /, compressing them
    first_pass: List[Token] = []
    current: Number = tokens[0]  # guaranteed number from the check above
    i = 1
    while i < len(tokens):
        op = tokens[i]
        if not isinstance(op, str):
            raise ValueError(f"Invalid token at position {i}: expected an operator")

        if i + 1 >= len(tokens):
            raise ValueError("Operator at end of expression")

        nxt = tokens[i + 1]
        if not isinstance(nxt, (int, float)):
            raise ValueError(f"Invalid token at position {i + 1}: expected a number")

        if op in ('*', '/'):
            current = apply_operator(current, nxt, op)
        elif op in ('+', '-'):
            first_pass.append(current)
            first_pass.append(op)
            current = nxt
        else:
            raise ValueError(f"Unsupported operator: {op}")

        i += 2

    first_pass.append(current)

    # Second pass: handle + and -
    result: Number = first_pass[0]  # should be a number
    j = 1
    while j < len(first_pass):
        op = first_pass[j]
        nxt = first_pass[j + 1]
        result = apply_operator(result, nxt, op)  # op should be '+' or '-'
        j += 2

    return result


def parse_and_calculate(expr: str) -> Number:
    """
    Parse, validate, and calculate the result of an arithmetic expression string.
    Supports digits, +, -, *, /, parentheses, and spaces.
    Applies correct operator precedence and handles unary + and -.
    Raises ValueError for invalid input.
    """
    if not validate_expression(expr):
        raise ValueError("Invalid characters in expression")

    # Tokenize into numbers and operators/parentheses
    raw_tokens = re.findall(r'\d+|[+\-*/()]', expr)
    tokens: List[Union[int, str]] = [int(t) if t.isdigit() else t for t in raw_tokens]

    if not tokens:
        raise ValueError("Empty or invalid expression")

    # Normalize tokens: handle unary + and - and basic validation
    normalized: List[Union[int, str]] = []
    expect_primary = True  # expecting a number or '(' or unary sign
    net_sign = 1  # accumulates unary +/- signs
    paren_balance = 0

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if isinstance(tok, int):
            if not expect_primary:
                raise ValueError("Missing operator between numbers")
            normalized.append(net_sign * tok)
            net_sign = 1
            expect_primary = False
            i += 1
            continue

        # tok is a string operator or parenthesis
        if tok in ('+', '-'):
            if expect_primary:
                # unary sign accumulation
                net_sign *= -1 if tok == '-' else 1
            else:
                # binary operator
                normalized.append(tok)
                expect_primary = True
                net_sign = 1
            i += 1
            continue

        if tok in ('*', '/'):
            if expect_primary:
                raise ValueError("Operator in invalid position")
            normalized.append(tok)
            expect_primary = True
            net_sign = 1
            i += 1
            continue

        if tok == '(':
            if not expect_primary:
                raise ValueError("Missing operator before '('")
            if net_sign == -1:
                # inject -1 * ( ... )
                normalized.append(-1)
                normalized.append('*')
                net_sign = 1
            normalized.append('(')
            paren_balance += 1
            expect_primary = True
            i += 1
            continue

        if tok == ')':
            if expect_primary:
                raise ValueError("Empty parentheses or operator before ')'")
            paren_balance -= 1
            if paren_balance < 0:
                raise ValueError("Mismatched parentheses")
            normalized.append(')')
            expect_primary = False
            i += 1
            continue

        # Should never reach here due to validation regex
        raise ValueError(f"Unsupported token: {tok}")

    if paren_balance != 0:
        raise ValueError("Mismatched parentheses")
    if expect_primary:
        raise ValueError("Expression cannot end with an operator")

    # Evaluate by resolving innermost parentheses first
    eval_tokens: List[Union[int, str]] = normalized[:]
    while True:
        try:
            l = len(eval_tokens) - 1 - eval_tokens[::-1].index('(')  # last '('
            # find matching ')'
            r = l + 1
            depth = 1
            while r < len(eval_tokens) and depth > 0:
                if eval_tokens[r] == '(':
                    depth += 1
                elif eval_tokens[r] == ')':
                    depth -= 1
                r += 1
            if depth != 0:
                raise ValueError("Mismatched parentheses")
            r -= 1  # r now at ')'
            inner = eval_tokens[l + 1:r]
            if not inner:
                raise ValueError("Empty parentheses")
            inner_result = calculate_with_precedence(inner)  # type: ignore[arg-type]
            # replace ( inner ) with inner_result
            eval_tokens = eval_tokens[:l] + [inner_result] + eval_tokens[r + 1:]
        except ValueError:
            # No '(' found
            break
        except Exception as e:
            # Propagate calculation errors as ValueError
            if isinstance(e, ValueError):
                raise
            raise

    # Final flat evaluation
    return calculate_with_precedence(eval_tokens)  # type: ignore[arg-type]
