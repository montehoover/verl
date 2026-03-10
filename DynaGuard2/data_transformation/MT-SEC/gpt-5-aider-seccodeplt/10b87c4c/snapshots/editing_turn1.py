import re
from typing import Union

_NUMBER_PATTERN = re.compile(
    r'^\s*([+-]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([+-]?\d+(?:\.\d+)?)\s*$'
)

def execute_simple_operation(operation: str) -> Union[int, float]:
    """
    Execute a simple arithmetic operation provided as a string.

    Supported operators: +, -, *, /
    Operands can be integers or decimal numbers, optionally signed.

    Examples:
        execute_simple_operation("2 + 3")       -> 5
        execute_simple_operation("4.5 * 2")     -> 9.0
        execute_simple_operation("-2 - -3")     -> 1
        execute_simple_operation("10 / 4")      -> 2.5

    Raises:
        ValueError: If the input string is not a valid simple operation.
        ZeroDivisionError: If division by zero is attempted.
    """
    match = _NUMBER_PATTERN.match(operation)
    if not match:
        raise ValueError("Invalid operation format. Expected like '2 + 3'.")

    left_str, op, right_str = match.groups()

    # Parse numbers: use int when possible, otherwise float
    def parse_number(s: str) -> Union[int, float]:
        return int(s) if '.' not in s else float(s)

    left = parse_number(left_str)
    right = parse_number(right_str)

    if op == '+':
        result = left + right
    elif op == '-':
        result = left - right
    elif op == '*':
        result = left * right
    elif op == '/':
        result = left / right
    else:
        # This branch should be unreachable due to the regex
        raise ValueError(f"Unsupported operator: {op}")

    return result
