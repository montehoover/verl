def execute_operation(operation: str) -> float:
    """
    Executes a simple arithmetic operation given as a string.

    Args:
        operation: A string representing a simple arithmetic operation,
                   e.g., '5 + 3', '10 / 2'.
                   Supports '+', '-', '*', '/'.

    Returns:
        The result of the operation as a float.

    Raises:
        ValueError: If the operation string is malformed, contains unsupported
                    operators, or if operands cannot be converted to numbers.
        ZeroDivisionError: If the operation involves division by zero.
    """
    parts = operation.split()
    if len(parts) != 3:
        raise ValueError(
            "Invalid operation format. Expected 'operand operator operand', "
            "e.g., '5 + 3'."
        )

    operand1_str, operator, operand2_str = parts

    try:
        operand1 = float(operand1_str)
        operand2 = float(operand2_str)
    except ValueError:
        raise ValueError(
            f"Invalid operands: '{operand1_str}', '{operand2_str}'. "
            "Operands must be numbers."
        )

    if operator == '+':
        return operand1 + operand2
    elif operator == '-':
        return operand1 - operand2
    elif operator == '*':
        return operand1 * operand2
    elif operator == '/':
        if operand2 == 0:
            raise ZeroDivisionError("Division by zero.")
        return operand1 / operand2
    else:
        raise ValueError(f"Unsupported operator: '{operator}'. "
                         "Supported operators are '+', '-', '*', '/'.")

if __name__ == '__main__':
    # Example usage and basic tests
    test_cases = {
        "5 + 3": 8.0,
        "10 - 2.5": 7.5,
        "4 * 2": 8.0,
        "10 / 2": 5.0,
        "7 / 0": "ZeroDivisionError",
        "5 & 3": "ValueError",
        "10 +": "ValueError",
        "a + 3": "ValueError",
        "3 + b": "ValueError",
        "10 / 4": 2.5
    }

    for op_str, expected in test_cases.items():
        print(f"Executing: '{op_str}'")
        try:
            result = execute_operation(op_str)
            if isinstance(expected, str): # Expected an error
                print(f"  ERROR: Expected {expected}, but got {result}")
            elif abs(result - expected) < 1e-9:
                print(f"  Success: {result}")
            else:
                print(f"  ERROR: Expected {expected}, got {result}")
        except (ValueError, ZeroDivisionError) as e:
            if isinstance(expected, str) and expected == type(e).__name__:
                print(f"  Success: Caught expected {type(e).__name__}: {e}")
            else:
                print(f"  ERROR: Unexpected {type(e).__name__}: {e}. Expected: {expected}")
        except Exception as e:
            print(f"  ERROR: An unexpected error occurred: {type(e).__name__}: {e}")
        print("-" * 20)
