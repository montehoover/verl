from typing import Union, Optional

def basic_calculate(num1_or_expr: Union[float, str],
                    num2_val: Optional[float] = None,
                    op_val: Optional[str] = None) -> float:
    """
    Performs basic arithmetic operations.
    Can be called in two ways:
    1. basic_calculate(expression_string: str) -> float
       e.g., basic_calculate("4 + 5")
    2. basic_calculate(num1: float, num2: float, operator: str) -> float
       e.g., basic_calculate(4, 5, '+')

    Args:
        num1_or_expr: Either a string expression like "num1 op num2" (e.g., "2.5 * 4")
                      or the first number (float).
        num2_val: The second number (float). Required if num1_or_expr is a number.
                  Should be None if num1_or_expr is a string expression.
        op_val: The operator string (e.g., '+', '-', '*', '/'). Required if
                num1_or_expr is a number. Should be None if num1_or_expr is a
                string expression.

    Returns:
        The result of the arithmetic operation as a float.

    Raises:
        ValueError: If the operator is invalid, division by zero is attempted,
                    the string expression is malformed, or numbers in the expression
                    are invalid.
        TypeError: If arguments are not provided in one of the two supported ways.
    """
    actual_num1: float
    actual_num2: float
    actual_operator: str

    if isinstance(num1_or_expr, str) and num2_val is None and op_val is None:
        # Mode 1: Called with a string expression "number operator number"
        parts = num1_or_expr.split()
        if len(parts) != 3:
            raise ValueError(
                "Invalid string expression format. Expected 'number operator number' "
                f"(e.g., '2 + 3'). Got: '{num1_or_expr}'"
            )
        try:
            actual_num1 = float(parts[0])
            actual_operator = parts[1]
            actual_num2 = float(parts[2])
        except ValueError:
            raise ValueError(
                f"Invalid number(s) in string expression: '{num1_or_expr}'. "
                "Numbers must be convertible to float."
            )
    elif isinstance(num1_or_expr, (int, float)) and \
         isinstance(num2_val, (int, float)) and \
         isinstance(op_val, str):
        # Mode 2: Called with three arguments (num1, num2, operator)
        actual_num1 = float(num1_or_expr)
        actual_num2 = float(num2_val)
        actual_operator = op_val
    else:
        raise TypeError(
            "Invalid arguments. Call with either a single string expression "
            "(e.g., '2 + 3') or three arguments (num1: float, num2: float, operator: str)."
        )

    # Core calculation logic
    if actual_operator == '+':
        return actual_num1 + actual_num2
    elif actual_operator == '-':
        return actual_num1 - actual_num2
    elif actual_operator == '*':
        return actual_num1 * actual_num2
    elif actual_operator == '/':
        if actual_num2 == 0:
            raise ValueError("Cannot divide by zero.")
        return actual_num1 / actual_num2
    else:
        raise ValueError(f"Invalid operator: '{actual_operator}'. Must be '+', '-', '*', or '/'.")
