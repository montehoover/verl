import re
from typing import List, Union

def sanitize_expression(expression: str) -> bool:
    """
    Return True if the expression contains only digits, +, -, *, /, and spaces.
    """
    if not isinstance(expression, str):
        return False
    return re.fullmatch(r'[0-9+\-*/ ]+', expression) is not None


def parse_expression(expression: str) -> List[Union[int, str]]:
    """
    Parse a sanitized arithmetic expression (containing digits, +, -, *, /, and spaces)
    and return a list of numbers and operators in Reverse Polish Notation (RPN),
    which reflects the order in which operations should be evaluated.

    Rules/assumptions:
    - Only binary operators +, -, *, / are supported.
    - Numbers are non-negative integers (no decimals, no unary plus/minus).
    - Expression must be properly formatted: it must start and end with a number,
      and operators and numbers must alternate.
    - Spaces are ignored.

    Raises:
        TypeError: If expression is not a string.
        ValueError: If the expression contains invalid characters or is improperly formatted.
    """
    if not isinstance(expression, str):
        raise TypeError("Expression must be a string")

    if not sanitize_expression(expression):
        raise ValueError("Expression contains invalid characters")

    # Tokenize: produce a flat list of ints and operator strings
    tokens: List[Union[int, str]] = []
    i = 0
    n = len(expression)

    while i < n:
        ch = expression[i]
        if ch == ' ':
            i += 1
            continue
        if ch.isdigit():
            start = i
            while i < n and expression[i].isdigit():
                i += 1
            tokens.append(int(expression[start:i]))
            continue
        if ch in "+-*/":
            tokens.append(ch)
            i += 1
            continue
        # Should not reach here due to sanitize_expression
        raise ValueError(f"Invalid character encountered: {ch!r}")

    if not tokens:
        raise ValueError("Empty expression")

    # Validate proper formatting: number (op number)* pattern, no leading/trailing operator
    if not isinstance(tokens[0], int) or not isinstance(tokens[-1], int):
        raise ValueError("Expression must start and end with a number")

    expect_number = True
    for tok in tokens:
        if expect_number:
            if isinstance(tok, int):
                expect_number = False
            else:
                raise ValueError("Invalid format: expected a number")
        else:
            if isinstance(tok, str) and tok in "+-*/":
                expect_number = True
            else:
                raise ValueError("Invalid format: expected an operator")

    if expect_number:
        # Last token was an operator
        raise ValueError("Expression cannot end with an operator")

    # Shunting-yard to produce RPN (evaluation order)
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    output: List[Union[int, str]] = []
    op_stack: List[str] = []

    for tok in tokens:
        if isinstance(tok, int):
            output.append(tok)
        else:
            while op_stack and precedence[op_stack[-1]] >= precedence[tok]:
                output.append(op_stack.pop())
            op_stack.append(tok)

    while op_stack:
        output.append(op_stack.pop())

    return output
