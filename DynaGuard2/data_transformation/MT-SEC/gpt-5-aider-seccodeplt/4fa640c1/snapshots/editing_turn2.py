from typing import Literal, Optional, Union, overload
import re

Operator = Literal['+', '-', '*', '/']


@overload
def basic_arithmetic(expression: str) -> float: ...
@overload
def basic_arithmetic(num1: float, num2: float, operator: Operator) -> float: ...


def basic_arithmetic(
    num1: Union[float, str],
    num2: Optional[float] = None,
    operator: Optional[Operator] = None
) -> float:
    """
    Perform a basic arithmetic operation.

    Usage 1 (explicit args):
        basic_arithmetic(5.0, 7.0, '+') -> 12.0

    Usage 2 (string expression):
        basic_arithmetic('5 + 7') -> 12.0

    Args:
        num1: First operand as float, or a string expression like '5 + 7'.
        num2: Second operand (required if num1 is float).
        operator: One of '+', '-', '*', '/' (required if num1 is float).

    Returns:
        The result as a float.

    Raises:
        ValueError: If inputs are invalid or operator is unsupported.
        ZeroDivisionError: If division by zero is attempted.
    """
    def _compute(a: float, b: float, op: Operator) -> float:
        if op == '+':
            return float(a + b)
        elif op == '-':
            return float(a - b)
        elif op == '*':
            return float(a * b)
        elif op == '/':
            if b == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            return float(a / b)
        else:
            raise ValueError(f"Unsupported operator: {op!r}. Expected one of '+', '-', '*', '/'.")

    # String expression mode
    if isinstance(num1, str) and num2 is None and operator is None:
        expr = num1
        match = re.match(
            r'^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([+\-*/])\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*$',
            expr
        )
        if not match:
            raise ValueError("Invalid expression format. Expected format like '5 + 7'.")
        a_str, op, b_str = match.groups()
        a = float(a_str)
        b = float(b_str)
        return _compute(a, b, op)  # type: ignore[arg-type]

    # Explicit args mode
    if isinstance(num1, (int, float)) and isinstance(num2, (int, float)) and operator in ('+', '-', '*', '/'):
        a = float(num1)
        b = float(num2)
        op = operator  # type: ignore[assignment]
        return _compute(a, b, op)  # type: ignore[arg-type]

    raise ValueError(
        "Invalid arguments. Provide either (num1: float, num2: float, operator: '+', '-', '*', '/') "
        "or a single expression string like '5 + 7'."
    )
