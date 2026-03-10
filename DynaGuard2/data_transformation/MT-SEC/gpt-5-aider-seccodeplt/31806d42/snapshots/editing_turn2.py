import re
from typing import Literal, Optional, Union, overload


float_re = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_expression_pattern = re.compile(rf'^\s*({float_re})\s*([+\-*/])\s*({float_re})\s*$')


@overload
def basic_calculator(num1: float, num2: float, operation: Literal['+', '-', '*', '/']) -> float: ...
@overload
def basic_calculator(expression: str) -> float: ...


def basic_calculator(
    num1_or_expression: Union[float, str],
    num2: Optional[float] = None,
    operation: Optional[Literal['+', '-', '*', '/']] = None
) -> float:
    """
    Perform a basic arithmetic operation.

    Two supported call forms:
    - basic_calculator(num1: float, num2: float, operation: '+', '-', '*', '/')
    - basic_calculator(expression: str)  # e.g., "3 + 4", "-2.5*-3", "6/2"

    Returns:
        The result of the arithmetic operation as a float.

    Raises:
        ValueError: If inputs are invalid or the operation is unsupported.
        ZeroDivisionError: If division by zero is attempted.
    """
    def _compute(a: float, b: float, op: str) -> float:
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
            raise ValueError("Invalid operation. Must be one of '+', '-', '*', '/'.")

    # Expression string mode
    if isinstance(num1_or_expression, str):
        if num2 is not None or operation is not None:
            raise ValueError("When passing an expression string, do not provide num2 or operation.")
        match = _expression_pattern.match(num1_or_expression)
        if not match:
            raise ValueError("Invalid expression format. Expected form like '3 + 4'.")
        a_str, op, b_str = match.groups()
        a = float(a_str)
        b = float(b_str)
        return _compute(a, b, op)

    # Numeric operands + explicit operation mode
    if num2 is None or operation is None:
        raise ValueError("When not using an expression string, provide num2 and operation.")
    return _compute(float(num1_or_expression), float(num2), operation)
