from typing import overload

@overload
def basic_calculate(expression: str) -> float: ...

@overload
def basic_calculate(num1: float, num2: float, operator: str) -> float: ...

def basic_calculate(*args) -> float:
    """
    Performs basic arithmetic operations.
    Can be called in two ways:
    1. With a single string expression: basic_calculate("num1 operator num2")
       Example: basic_calculate("4.5 + 5")
    2. With three arguments: basic_calculate(num1, num2, operator)
       Example: basic_calculate(4.5, 5.0, "+")

    Args:
        *args: Variable length argument list.
               Expects either:
               - A single string: `expression` (e.g., "4 + 5")
               - Three arguments: `num1` (float-convertible), `num2` (float-convertible),
                 `operator` (str: '+', '-', '*', '/')

    Returns:
        float: The result of the arithmetic operation.

    Raises:
        TypeError: If the arguments are not provided in one of the two supported ways,
                   or if argument types are incorrect for the 3-argument call.
        ValueError: If the expression string is malformed, contains invalid numbers,
                    the operator is invalid, or if division by zero is attempted.
    """
    num1: float
    num2: float
    operator: str

    if len(args) == 1 and isinstance(args[0], str):
        expression: str = args[0]
        parts = expression.split()
        if len(parts) != 3:
            raise ValueError(
                "Expression string must be in the format 'operand operator operand', e.g., '4 + 5'. "
                f"Received: '{expression}'"
            )
        try:
            num1_str, op_str, num2_str = parts
            num1 = float(num1_str)
            operator = op_str
            num2 = float(num2_str)
        except ValueError: # Catches float conversion errors
            raise ValueError(
                f"Invalid number format in expression string: '{expression}'. "
                "Ensure operands are valid numbers."
            )
    elif len(args) == 3:
        # Type checking for the 3-argument case
        if not (isinstance(args[0], (int, float)) and \
                isinstance(args[1], (int, float)) and \
                isinstance(args[2], str)):
            raise TypeError(
                "For 3 arguments, expected num1 (float-convertible), num2 (float-convertible), operator (str). "
                f"Received types: ({type(args[0])}, {type(args[1])}, {type(args[2])})"
            )
        try:
            num1 = float(args[0])
            num2 = float(args[1])
        except ValueError: # Should not happen if previous check passes for int/float, but good for safety
            raise TypeError(
                "For 3 arguments, num1 and num2 must be convertible to float."
            )
        operator = args[2]
    else:
        raise TypeError(
            "basic_calculate accepts either a single string expression (e.g., '4 + 5') "
            "or three arguments: num1 (float), num2 (float), operator (str)."
        )

    # Core calculation logic
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ValueError("Cannot divide by zero")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: '{operator}'. Must be '+', '-', '*', or '/'.")
