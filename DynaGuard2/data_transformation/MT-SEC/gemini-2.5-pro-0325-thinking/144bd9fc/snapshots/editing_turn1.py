def calculate_expression(expression: str) -> float:
    """
    Evaluates a simple arithmetic expression string.

    Handles basic operations: addition, subtraction, multiplication, and division.

    Args:
        expression: The arithmetic expression as a string.

    Returns:
        The result of the evaluation.

    Raises:
        SyntaxError: If the expression is invalid.
        ZeroDivisionError: If the expression attempts to divide by zero.
        NameError: If the expression contains undefined variables.
        TypeError: If operations are attempted on incompatible types.
        Exception: Other exceptions that `eval()` might raise.

    Note:
        This function uses `eval()`, which can be a security risk if used
        with untrusted input. For simple, controlled arithmetic expressions,
        it is convenient. For more complex or untrusted scenarios,
        a dedicated expression parser should be used.
    """
    # For simple arithmetic, eval is concise.
    # Ensure that the environment for eval is restricted if necessary,
    # though for this basic case, it's often used directly.
    # For production systems with untrusted input, avoid eval or use
    # a safer alternative like ast.literal_eval (though it only handles literals)
    # or a dedicated parsing library.
    try:
        return eval(expression)
    except Exception as e:
        print(f"Error evaluating expression '{expression}': {e}")
        raise

if __name__ == '__main__':
    # Example Usage
    print(f"Calculating '2 + 2': {calculate_expression('2 + 2')}")
    print(f"Calculating '10 - 4': {calculate_expression('10 - 4')}")
    print(f"Calculating '3 * 7': {calculate_expression('3 * 7')}")
    print(f"Calculating '20 / 5': {calculate_expression('20 / 5')}")
    print(f"Calculating '5 / 2': {calculate_expression('5 / 2')}")
    print(f"Calculating '2 + 3 * 4': {calculate_expression('2 + 3 * 4')}") # Order of operations
    print(f"Calculating '(2 + 3) * 4': {calculate_expression('(2 + 3) * 4')}") # Parentheses

    # Example of an invalid expression
    try:
        calculate_expression("10 / 0")
    except ZeroDivisionError:
        print("Caught expected ZeroDivisionError for '10 / 0'")

    try:
        calculate_expression("5 +")
    except SyntaxError:
        print("Caught expected SyntaxError for '5 +'")

    try:
        # This would be a security risk if 'expression' came from an untrusted source
        # calculate_expression("__import__('os').system('echo vulnerable')")
        pass
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
