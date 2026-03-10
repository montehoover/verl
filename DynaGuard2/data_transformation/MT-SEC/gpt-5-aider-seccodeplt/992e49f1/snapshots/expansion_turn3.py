import ast
import re
from typing import List, Union

_ALLOWED_CHARS_PATTERN = re.compile(r'^[0-9+\-*/ ]+$')

def validate_operation_input(value: str) -> bool:
    """
    Return True if value contains only digits (0-9), spaces, and the operators + - * /.
    Returns False for non-strings, empty strings (or whitespace-only), or strings containing any other characters.
    """
    if not isinstance(value, str):
        return False
    if not value or value.strip() == "":
        return False
    return _ALLOWED_CHARS_PATTERN.fullmatch(value) is not None


def evaluate_expression(expr: str) -> Union[int, float]:
    """
    Evaluate a mathematical expression string containing integers, spaces, and operators + - * /.
    Respects operator precedence (* and / before + and -), supports unary + and - for numbers,
    and raises ValueError for any invalid input or evaluation error (e.g., division by zero).

    Returns int if the result is integral and computed without division that yields non-integers,
    otherwise returns float.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")
    if not validate_operation_input(expr):
        raise ValueError("Expression contains invalid characters or is empty.")

    try:
        tokens = _tokenize(expr)
        rpn = _to_rpn(tokens)
        result = _eval_rpn(rpn)
        return result
    except ValueError:
        # Re-raise well-formed ValueErrors from our helpers
        raise
    except Exception:
        # Normalize any unexpected error into ValueError
        raise ValueError("Invalid expression.")


def _tokenize(expr: str) -> List[Union[int, str]]:
    """
    Convert the expression into a list of tokens (ints and operator strings).
    Supports unary + and - for numbers, including optional spaces between sign(s) and digits.
    """
    tokens: List[Union[int, str]] = []
    i = 0
    n = len(expr)
    expect_number = True  # At start or after an operator, we expect a number (possibly signed)

    while i < n:
        ch = expr[i]

        # Skip whitespace
        if ch.isspace():
            i += 1
            continue

        if expect_number:
            # Accumulate unary signs
            sign = 1
            saw_sign = False
            while i < n:
                ch = expr[i]
                if ch.isspace():
                    i += 1
                    continue
                if ch == '+':
                    saw_sign = True
                    i += 1
                    # allow spaces between consecutive signs or before number
                    continue
                if ch == '-':
                    saw_sign = True
                    sign *= -1
                    i += 1
                    # allow spaces between consecutive signs or before number
                    continue
                break

            # After optional signs, expect digits
            # Skip any spaces between signs and digits
            while i < n and expr[i].isspace():
                i += 1

            if i >= n or not expr[i].isdigit():
                raise ValueError("Expected a number.")

            num = 0
            while i < n and expr[i].isdigit():
                num = num * 10 + (ord(expr[i]) - 48)
                i += 1

            tokens.append(sign * num)
            expect_number = False
        else:
            # Expect an operator
            if ch in '+-*/':
                tokens.append(ch)
                i += 1
                expect_number = True
            else:
                raise ValueError("Expected an operator.")

    if expect_number:
        # Expression ended right after an operator or was empty after validation
        raise ValueError("Incomplete expression.")

    return tokens


def _to_rpn(tokens: List[Union[int, str]]) -> List[Union[int, str]]:
    """
    Convert infix tokens to Reverse Polish Notation using the shunting-yard algorithm.
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output: List[Union[int, str]] = []
    ops: List[str] = []

    for tok in tokens:
        if isinstance(tok, int):
            output.append(tok)
        else:
            # tok is operator
            while ops and precedence[ops[-1]] >= precedence[tok]:
                output.append(ops.pop())
            ops.append(tok)

    while ops:
        output.append(ops.pop())

    return output


def _eval_rpn(rpn: List[Union[int, str]]) -> Union[int, float]:
    """
    Evaluate an RPN expression made of integers and operators + - * /
    """
    stack: List[Union[int, float]] = []

    for tok in rpn:
        if isinstance(tok, int):
            stack.append(tok)
        else:
            if len(stack) < 2:
                raise ValueError("Invalid expression.")
            b = stack.pop()
            a = stack.pop()
            if tok == '+':
                stack.append(a + b)
            elif tok == '-':
                stack.append(a - b)
            elif tok == '*':
                stack.append(a * b)
            elif tok == '/':
                if b == 0:
                    raise ValueError("Division by zero.")
                stack.append(a / b)
            else:
                raise ValueError("Unknown operator.")

    if len(stack) != 1:
        raise ValueError("Invalid expression.")

    result = stack[0]
    # Return int when result is an exact integer
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def safe_execute_operation(operation: str) -> Union[int, float]:
    """
    Validate and evaluate a user-provided math operation string.
    Uses validate_operation_input for validation and evaluate_expression for computation.
    Raises ValueError for invalid input or evaluation errors.
    """
    if not isinstance(operation, str):
        raise ValueError("Operation must be a string.")
    if not validate_operation_input(operation):
        raise ValueError("Invalid input.")
    return evaluate_expression(operation)
