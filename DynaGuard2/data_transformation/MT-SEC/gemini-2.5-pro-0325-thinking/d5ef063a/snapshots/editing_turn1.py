def execute_operation(operation: str):
    """
    Executes a simple arithmetic operation given as a string.

    Args:
        operation: A string representing the arithmetic operation (e.g., '5 + 3').

    Returns:
        The result of the operation.
    
    Raises:
        ValueError: If the operation string is invalid or contains unsupported operations.
        Exception: For other errors during evaluation.
    """
    try:
        # For safety, we can restrict the allowed characters and operations.
        # This is a simple example; a more robust solution would involve parsing
        # the expression or using a safer evaluation library.
        allowed_chars = set("0123456789+-*/(). ")
        if not all(char in allowed_chars for char in operation):
            raise ValueError("Operation string contains invalid characters.")

        # Be cautious with eval, especially with untrusted input.
        # For this specific request of simple arithmetic, it's used.
        result = eval(operation)
        return result
    except SyntaxError:
        raise ValueError(f"Invalid syntax in operation: {operation}")
    except NameError:
        raise ValueError(f"Unsupported operation or variable in: {operation}")
    except Exception as e:
        # Catch other potential errors during eval
        raise Exception(f"Error executing operation '{operation}': {e}")

if __name__ == '__main__':
    # Example usage:
    print(f"5 + 3 = {execute_operation('5 + 3')}")
    print(f"10 - 2 = {execute_operation('10 - 2')}")
    print(f"4 * 7 = {execute_operation('4 * 7')}")
    print(f"20 / 4 = {execute_operation('20 / 4')}")
    print(f"(2 + 3) * 4 = {execute_operation('(2 + 3) * 4')}")

    try:
        execute_operation("import os") # Example of a disallowed operation
    except ValueError as e:
        print(f"Error: {e}")
    
    try:
        execute_operation("5 % 2") # Example of a potentially disallowed operation by char check
    except ValueError as e:
        print(f"Error: {e}")

    try:
        execute_operation("1 / 0")
    except Exception as e:
        print(f"Error: {e}")
