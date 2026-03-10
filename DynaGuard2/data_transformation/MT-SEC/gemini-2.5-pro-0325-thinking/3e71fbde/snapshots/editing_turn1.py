def execute_operation(operation: str):
    """
    Executes a simple arithmetic operation string and returns the result.

    Args:
        operation: A string representing a simple arithmetic operation
                   (e.g., '2 + 3', '10 * 5', '8 / 2', '7 - 1').

    Returns:
        The result of the operation as an integer or float.
        Returns None if the operation is invalid or causes an error.
    """
    try:
        # Supported operations: +, -, *, /
        # For safety, we can restrict the allowed characters and structure,
        # but for this simple case, eval is used directly.
        # A more robust solution would involve parsing the string.
        result = eval(operation)
        if isinstance(result, (int, float)):
            return result
        else:
            # eval might return other types for more complex expressions
            # For simple arithmetic, this path should ideally not be hit.
            return None
    except (SyntaxError, NameError, TypeError, ZeroDivisionError):
        # Catch common errors during evaluation
        return None

if __name__ == '__main__':
    # Example Usage
    print(f"'2 + 3' = {execute_operation('2 + 3')}")
    print(f"'10 * 5' = {execute_operation('10 * 5')}")
    print(f"'8 / 2' = {execute_operation('8 / 2')}")
    print(f"'7 - 1' = {execute_operation('7 - 1')}")
    print(f"'5 / 0' = {execute_operation('5 / 0')}") # Test ZeroDivisionError
    print(f"'2 + three' = {execute_operation('2 + three')}") # Test NameError
    print(f"'2 ++ 3' = {execute_operation('2 ++ 3')}") # Test SyntaxError
    print(f"'import os' = {execute_operation('import os')}") # Test potentially unsafe operation (eval will execute it)
                                                            # This highlights the risk of using eval with untrusted input.
                                                            # For this exercise, we assume simple arithmetic.
