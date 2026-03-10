def evaluate_expression(expression: str):
    """
    Evaluates a simple Python expression given as a string and returns the result.

    Args:
        expression: The string containing the Python expression.

    Returns:
        The result of the evaluated expression.
    """
    try:
        return eval(expression)
    except Exception as e:
        # It's good practice to handle potential errors during evaluation.
        # For now, we'll re-raise the exception, but you might want to
        # return a specific error message or value depending on your needs.
        print(f"Error evaluating expression '{expression}': {e}")
        raise

if __name__ == '__main__':
    # Example usage:
    expr1 = "1 + 1"
    expr2 = "5 * (3 - 1)"
    expr3 = "len('hello')"
    expr4 = "10 / 0" # Example of an expression that will raise an error

    print(f"'{expr1}' => {evaluate_expression(expr1)}")
    print(f"'{expr2}' => {evaluate_expression(expr2)}")
    print(f"'{expr3}' => {evaluate_expression(expr3)}")

    try:
        print(f"'{expr4}' => {evaluate_expression(expr4)}")
    except ZeroDivisionError:
        print(f"Attempted to evaluate '{expr4}', which resulted in a ZeroDivisionError as expected.")
    except Exception as e:
        print(f"An unexpected error occurred while evaluating '{expr4}': {e}")
