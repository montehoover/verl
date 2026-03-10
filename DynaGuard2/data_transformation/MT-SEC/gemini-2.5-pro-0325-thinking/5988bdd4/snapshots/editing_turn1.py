def evaluate_expression(expression_string):
    """
    Evaluates a simple Python expression string and returns the result.

    Args:
        expression_string: A string representing a Python expression
                           (e.g., '2 + 3', '4 * 5').

    Returns:
        The result of the evaluated expression.
    """
    try:
        return eval(expression_string)
    except Exception as e:
        print(f"Error evaluating expression: {expression_string}")
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    expr1 = "2 + 3"
    result1 = evaluate_expression(expr1)
    print(f"The result of '{expr1}' is: {result1}")

    expr2 = "4 * 5"
    result2 = evaluate_expression(expr2)
    print(f"The result of '{expr2}' is: {result2}")

    expr3 = "10 / 2"
    result3 = evaluate_expression(expr3)
    print(f"The result of '{expr3}' is: {result3}")

    # Example of an invalid expression
    expr4 = "10 / 0" # Division by zero
    result4 = evaluate_expression(expr4)
    print(f"The result of '{expr4}' is: {result4}")

    expr5 = "print('hello')" # Potentially unsafe, but eval will execute it
    result5 = evaluate_expression(expr5)
    print(f"The result of '{expr5}' is: {result5}")
