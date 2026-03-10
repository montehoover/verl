def execute_operation(operation_str: str):
    """
    Executes a simple arithmetic operation from a string.

    Args:
        operation_str: A string representing a simple arithmetic
                       operation, e.g., '2 + 3', '10 * 5'.

    Returns:
        The result of the arithmetic operation.

    Raises:
        SyntaxError: If the operation_str is not a valid Python expression.
        NameError: If the operation_str contains undefined variables.
        TypeError: If operations are attempted on incompatible types.
        Exception: For other evaluation errors.
    """
    # For security reasons, eval() should only be used with trusted input.
    # This function assumes the input string contains simple arithmetic.
    try:
        result = eval(operation_str)
        return result
    except (SyntaxError, NameError, TypeError) as e:
        # Re-raise specific, common errors for clarity
        raise e
    except Exception as e:
        # Catch any other eval-related errors
        raise Exception(f"Error evaluating operation '{operation_str}': {e}")

if __name__ == '__main__':
    # Example usage:
    print(f"Result of '2 + 3': {execute_operation('2 + 3')}")
    print(f"Result of '10 - 4': {execute_operation('10 - 4')}")
    print(f"Result of '6 * 7': {execute_operation('6 * 7')}")
    print(f"Result of '8 / 2': {execute_operation('8 / 2')}")
    print(f"Result of '2 ** 3': {execute_operation('2 ** 3')}")

    try:
        execute_operation("1 / 0")
    except ZeroDivisionError as e:
        print(f"Error: {e}")

    try:
        execute_operation("import os") # Example of potentially unsafe input
    except Exception as e:
        print(f"Error with unsafe input: {e}")

    try:
        execute_operation("a + 3") # Example of NameError
    except NameError as e:
        print(f"Error: {e}")

    try:
        execute_operation("2 +") # Example of SyntaxError
    except SyntaxError as e:
        print(f"Error: {e}")
