import re

def basic_arithmetic(num1: float = None, num2: float = None, operator: str = None, expression: str = None) -> float:
    """
    Performs basic arithmetic operations.
    Can accept two numbers and an operator, or a single string expression.

    Args:
        num1: The first number (float). Required if 'expression' is not provided.
        num2: The second number (float). Required if 'expression' is not provided.
        operator: The operation to perform. Must be one of '+', '-', '*', or '/'.
                  Required if 'expression' is not provided.
        expression: A string representing a simple arithmetic operation, e.g., "5 + 7".
                    If provided, 'num1', 'num2', and 'operator' are ignored.

    Returns:
        The result of the arithmetic operation (float).

    Raises:
        ValueError: If the operator is invalid, division by zero is attempted,
                    or the expression string is malformed.
    """
    if expression:
        # Try to parse the expression string
        # Regex to find two numbers and an operator, allowing for spaces
        match = re.fullmatch(r"\s*(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*", expression)
        if not match:
            raise ValueError(f"Invalid expression format: '{expression}'. Expected format like 'num1 operator num2'.")
        
        try:
            num1_str, op_str, num2_str = match.groups()
            num1_val = float(num1_str)
            num2_val = float(num2_str)
        except ValueError:
            raise ValueError(f"Invalid numbers in expression: '{expression}'.")

        # Use the parsed values
        _num1, _num2, _operator = num1_val, num2_val, op_str
    elif num1 is not None and num2 is not None and operator is not None:
        _num1, _num2, _operator = num1, num2, operator
    else:
        raise ValueError("Either 'expression' or all of 'num1', 'num2', and 'operator' must be provided.")

    if _operator == '+':
        return _num1 + _num2
    elif _operator == '-':
        return _num1 - _num2
    elif _operator == '*':
        return _num1 * _num2
    elif _operator == '/':
        if _num2 == 0:
            raise ValueError("Cannot divide by zero")
        return _num1 / _num2
    else:
        raise ValueError(f"Invalid operator: '{_operator}'. Must be '+', '-', '*', or '/'.")
