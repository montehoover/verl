from typing import Union, Optional

def basic_calculator(arg1: Union[str, float, int], arg2: Optional[float] = None, arg3: Optional[str] = None) -> float:
    """
    Performs basic arithmetic operations.
    Can be called in two ways:
    1. With a single string expression: basic_calculator("num1 operator num2")
       Example: basic_calculator("7 + 3")
    2. With three arguments: basic_calculator(num1: float, num2: float, operator: str)
       Example: basic_calculator(7.0, 3.0, '+')

    Args:
        arg1: Either a string expression "number operator number" (e.g., "7 + 3"),
              or the first number (float or int).
        arg2: The second number (float). Required and used only if arg1 is a number.
        arg3: The operator string ('+', '-', '*', '/'). Required and used only if arg1 is a number.

    Returns:
        The result of the arithmetic operation as a float.

    Raises:
        ValueError: If the input string is malformed, numbers are invalid,
                    operator is invalid, or if division by zero is attempted.
        TypeError: If arguments are missing for the numeric input mode,
                   or if unexpected arguments are provided for string input mode,
                   or if arg1 is not a string or number.
    """
    _num1: float
    _num2: float
    _operator: str

    if isinstance(arg1, str):
        if arg2 is not None or arg3 is not None:
            raise TypeError("When providing a string expression, arg2 and arg3 must not be provided.")
        
        parts = arg1.split()
        if len(parts) != 3:
            raise ValueError(
                "Invalid string expression format. Expected 'number operator number' (e.g., '7 + 3')."
            )
        
        try:
            _num1 = float(parts[0])
            _num2 = float(parts[2])
        except ValueError:
            raise ValueError(
                f"Invalid numbers in string expression: '{parts[0]}', '{parts[2]}'. Numbers must be convertible to float."
            )
        
        _operator = parts[1]

    elif isinstance(arg1, (float, int)):
        if arg2 is None or arg3 is None:
            raise TypeError("For numeric input (arg1 is a number), arg2 (second number) and arg3 (operator) are required.")
        if not isinstance(arg2, (float, int)):
            raise TypeError(f"arg2 must be a number (float or int), got {type(arg2)}.")
        if not isinstance(arg3, str):
             raise TypeError(f"arg3 must be a string operator, got {type(arg3)}.")

        _num1 = float(arg1)
        _num2 = float(arg2)
        _operator = arg3
    else:
        raise TypeError("arg1 must be a string expression or a number (float or int).")

    # Perform the calculation
    if _operator == '+':
        return _num1 + _num2
    elif _operator == '-':
        return _num1 - _num2
    elif _operator == '*':
        return _num1 * _num2
    elif _operator == '/':
        if _num2 == 0:
            raise ValueError("Cannot divide by zero.")
        return _num1 / _num2
    else:
        raise ValueError(f"Invalid operator: '{_operator}'. Must be one of '+', '-', '*', or '/'.")
