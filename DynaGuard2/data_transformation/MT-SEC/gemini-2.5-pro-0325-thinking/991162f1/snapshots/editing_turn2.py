def calculate_expression(expression: str) -> str:
    """
    Calculates the result of a basic arithmetic expression string.

    Args:
        expression: A string containing a basic arithmetic expression
                    (e.g., "2 + 3 * 4", "2 ** 3 + (4 * 5)").
                    Supports addition, subtraction, multiplication, division,
                    exponentiation, and parentheses for order of operations.

    Returns:
        The calculated result as a string.

    Raises:
        SyntaxError: If the expression is invalid.
        ZeroDivisionError: If the expression attempts to divide by zero.
        Exception: For other evaluation errors.
    """
    try:
        # WARNING: Using eval() can be dangerous if the input string is not
        # strictly controlled, as it can execute arbitrary code.
        # For this basic case, we assume the input is a simple arithmetic expression.
        result = eval(expression)
        return str(result)
    except (SyntaxError, ZeroDivisionError) as e:
        # Re-raise specific, expected errors
        raise e
    except Exception as e:
        # Handle other potential errors during evaluation
        raise Exception(f"Error evaluating expression: {e}")

if __name__ == '__main__':
    # Example usage:
    expressions = [
        "2 + 3",
        "10 - 4",
        "5 * 6",
        "20 / 4",
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "2 ** 3",
        "2 ** 3 + (4 * 5)",
        "(10 + 5) / 3 - 2**2",
        "10 / 0",  # Example of ZeroDivisionError
        "5 + ",    # Example of SyntaxError
        "2 ** (3 + 1)",
    ]

    for expr in expressions:
        try:
            result = calculate_expression(expr)
            print(f"Expression: '{expr}', Result: {result}")
        except Exception as e:
            print(f"Expression: '{expr}', Error: {e}")

    print(calculate_expression("3.14 * 2"))
    print(calculate_expression("100 / 2.5"))
