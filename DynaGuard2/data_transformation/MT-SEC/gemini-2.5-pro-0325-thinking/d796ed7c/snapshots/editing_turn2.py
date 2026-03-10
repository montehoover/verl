from typing import Union, Optional

def basic_calculator(arg1: Union[float, str], arg2: Optional[float] = None, arg3: Optional[str] = None) -> float:
    """
    Performs basic arithmetic operations.
    Can be called in two ways:
    1. With a single string expression: basic_calculator("num1 operator num2")
       Example: basic_calculator("5 + 7")
    2. With three arguments: basic_calculator(num1, num2, operator)
       Example: basic_calculator(5.0, 7.0, '+')

    Args:
        arg1: Either a string expression (e.g., "5 + 7") or the first number (float).
        arg2: The second number (float). Optional if arg1 is an expression string, otherwise required.
        arg3: The operator string ('+', '-', '*', '/'). Optional if arg1 is an expression string, otherwise required.

    Returns:
        The result of the arithmetic operation as a float.

    Raises:
        ValueError: If the expression string is malformed, numbers are invalid,
                    operator is invalid, or if division by zero is attempted.
        TypeError: If arguments are of incorrect types or an invalid combination.
    """
    num1: float
    num2: float
    operator: str

    if isinstance(arg1, str) and arg2 is None and arg3 is None:
        # Mode 1: Parse string expression "number operator number"
        parts = arg1.split()
        if len(parts) != 3:
            raise ValueError(
                f"Invalid expression format: '{arg1}'. Expected 'number operator number'."
            )
        
        num1_str, op_str, num2_str = parts
        
        try:
            num1 = float(num1_str)
        except ValueError:
            raise ValueError(f"Invalid first number in expression: '{num1_str}'")
        
        try:
            num2 = float(num2_str)
        except ValueError:
            raise ValueError(f"Invalid second number in expression: '{num2_str}'")
            
        operator = op_str

    elif isinstance(arg1, (int, float)) and isinstance(arg2, (int, float)) and isinstance(arg3, str):
        # Mode 2: num1, num2, operator arguments
        num1 = float(arg1) # Ensure num1 is float even if int is passed
        num2 = float(arg2) # Ensure num2 is float even if int is passed
        operator = arg3
    else:
        raise TypeError(
            "Invalid arguments. Call with a single expression string (e.g., '5 + 7') "
            "or with three arguments: basic_calculator(num1: float, num2: float, operator: str)."
        )

    # Validate operator
    if operator not in ['+', '-', '*', '/']:
        raise ValueError(f"Invalid operator: '{operator}'. Must be '+', '-', '*', or '/'.")

    # Perform calculation
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
    # No else needed here as operator is validated to be one of the above
