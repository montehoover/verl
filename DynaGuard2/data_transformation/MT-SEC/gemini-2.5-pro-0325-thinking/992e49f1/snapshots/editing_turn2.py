import re
from typing import Union, Optional

def basic_calculate(num1: Union[float, str], num2: Optional[float] = None, operator: Optional[str] = None) -> float:
    """
    Performs basic arithmetic operations.
    Can be called in two ways:
    1. basic_calculate(num1: float, num2: float, operator: str)
    2. basic_calculate(expression: str) where expression is like "5 + 3"

    Args:
        num1: The first number or an arithmetic expression string.
        num2: The second number (optional if num1 is an expression).
        operator: The operation to perform ('+', '-', '*', '/') (optional if num1 is an expression).

    Returns:
        The result of the arithmetic operation.

    Raises:
        ValueError: If the operator is invalid, division by zero is attempted,
                    or the expression string is malformed.
    """
    if isinstance(num1, str):
        # Try to parse the expression string
        parts = num1.split()
        if len(parts) != 3:
            raise ValueError("Invalid expression string format. Expected 'number operator number', e.g., '5 + 3'")
        
        try:
            val1 = float(parts[0])
            op = parts[1]
            val2 = float(parts[2])
        except ValueError:
            raise ValueError("Invalid numbers in expression string.")
        
        # Assign parsed values to num1, num2, operator for existing logic
        num1_calc, num2_calc, op_calc = val1, val2, op
    elif num2 is not None and operator is not None:
        num1_calc, num2_calc, op_calc = num1, num2, operator
    else:
        raise ValueError("Invalid arguments. Provide num1, num2, and operator, or a single expression string.")

    if op_calc == '+':
        return float(num1_calc + num2_calc)
    elif op_calc == '-':
        return float(num1_calc - num2_calc)
    elif op_calc == '*':
        return float(num1_calc * num2_calc)
    elif op_calc == '/':
        if num2_calc == 0:
            raise ValueError("Cannot divide by zero")
        return float(num1_calc / num2_calc)
    else:
        raise ValueError("Invalid operator. Must be one of '+', '-', '*', '/'")
