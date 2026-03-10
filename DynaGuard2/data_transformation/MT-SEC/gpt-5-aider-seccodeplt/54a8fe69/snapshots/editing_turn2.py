import re
from typing import Literal, Optional, Tuple, Union

def _apply_operation(a: float, b: float, op: str) -> float:
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
    raise ValueError(f"Unsupported operator: {op}. Use one of '+', '-', '*', '/'.")

def _parse_expression(expr: str) -> Tuple[float, float, str]:
    """
    Parse a simple arithmetic expression like '7 + 3' or '-2.5 * .4'
    and return (num1, num2, operator).
    """
    if not isinstance(expr, str):
        raise TypeError("Expression must be a string.")

    pattern = r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([+\-*/])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$'
    match = re.match(pattern, expr)
    if not match:
        raise ValueError("Invalid expression format. Expected format like '7 + 3'.")

    a_str, op, b_str = match.groups()
    return float(a_str), float(b_str), op

def basic_calculator(
    num1: Union[float, str],
    num2: Optional[float] = None,
    operator: Optional[Literal['+', '-', '*', '/']] = None
) -> float:
    """
    Perform a basic arithmetic operation.

    Usage:
    - Three-argument form: basic_calculator(7.0, 3.0, '+')
    - Single-string form:  basic_calculator('7 + 3')

    Returns:
        The result as a float.

    Raises:
        ValueError: If operator is not supported or input is invalid.
        ZeroDivisionError: If division by zero is attempted.
    """
    # Single-string expression form
    if isinstance(num1, str) and num2 is None and operator is None:
        a, b, op = _parse_expression(num1)
        return _apply_operation(a, b, op)

    # Three-argument form
    if isinstance(num1, (int, float)) and isinstance(num2, (int, float)) and operator in ('+', '-', '*', '/'):
        return _apply_operation(float(num1), float(num2), operator)  # type: ignore[arg-type]

    raise ValueError("Invalid arguments. Provide either (num1: float, num2: float, operator: '+', '-', '*', '/') or a single expression string like '7 + 3'.")
