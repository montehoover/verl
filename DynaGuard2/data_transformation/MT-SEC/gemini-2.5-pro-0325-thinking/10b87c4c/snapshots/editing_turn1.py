def execute_simple_operation(operation: str):
    """
    Executes a simple arithmetic operation given as a string.

    Args:
        operation: A string representing a simple arithmetic operation
                   (e.g., '2 + 3', '10 * 5', '8 / 2', '7 - 1').

    Returns:
        The result of the operation as an integer or float.

    Raises:
        SyntaxError: If the operation string is not a valid Python expression.
        NameError: If the operation string contains undefined variables.
        TypeError: If the operation involves incompatible types.
        ZeroDivisionError: If the operation involves division by zero.
        Exception: For other potential errors during evaluation.
    """
    try:
        # Be cautious with eval if the input string is not trusted.
        # For simple arithmetic from a controlled source, it's generally okay.
        result = eval(operation)
        if not isinstance(result, (int, float)):
            raise ValueError("Operation did not result in a number.")
        return result
    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
        # Re-raise specific, common errors for clarity
        raise e
    except Exception as e:
        # Catch any other exceptions during eval
        raise Exception(f"Error evaluating operation '{operation}': {e}")

if __name__ == '__main__':
    # Example Usage
    operations = [
        "2 + 3",
        "10 - 4",
        "5 * 6",
        "8 / 2",
        "2.5 * 4",
        "10 / 3",
        "7 % 3",
        "2 ** 3"
    ]

    for op_str in operations:
        try:
            result = execute_simple_operation(op_str)
            print(f"'{op_str}' = {result}")
        except Exception as e:
            print(f"Error executing '{op_str}': {e}")

    # Example of an invalid operation
    invalid_ops = [
        "2 +",
        "hello * 3",
        "10 / 0",
        "import os" # Demonstrating eval's potential danger if not restricted
    ]
    for op_str in invalid_ops:
        try:
            result = execute_simple_operation(op_str)
            print(f"'{op_str}' = {result}")
        except Exception as e:
            print(f"Error executing '{op_str}': {e}")
