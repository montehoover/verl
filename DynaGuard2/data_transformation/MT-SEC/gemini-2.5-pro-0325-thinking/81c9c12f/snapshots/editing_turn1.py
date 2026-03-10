def evaluate_expression(expression_string):
    """
    Evaluates a simple arithmetic expression string.

    Args:
        expression_string: The arithmetic expression (e.g., "2 + 3 * 4").

    Returns:
        The numerical result of the expression.

    Raises:
        SyntaxError: If the expression is invalid.
        NameError: If the expression contains invalid names.
        TypeError: If operations are attempted on incompatible types.
        ZeroDivisionError: If division by zero is attempted.
    """
    try:
        # WARNING: Using eval() can be a security risk if the input string
        # comes from an untrusted source. For simple, controlled arithmetic
        # expressions, it's generally acceptable.
        result = eval(expression_string)
        return result
    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
        print(f"Error evaluating expression: {e}")
        raise

if __name__ == '__main__':
    # Example usage:
    expressions = [
        "2 + 3",
        "10 - 4 * 2",
        "5 * (3 + 1)",
        "10 / 2",
        "7 + 2 * 3 - 1",
        "100 / (2 * 5)",
        "3.5 * 2 + 1.0",
    ]

    for expr in expressions:
        try:
            print(f"Expression: {expr} = {evaluate_expression(expr)}")
        except Exception:
            pass

    # Example of an invalid expression
    invalid_expr = "5 + "
    print(f"\nTrying invalid expression: {invalid_expr}")
    try:
        evaluate_expression(invalid_expr)
    except SyntaxError as e:
        print(f"Caught expected error: {e}")

    # Example of division by zero
    div_by_zero_expr = "10 / 0"
    print(f"\nTrying division by zero: {div_by_zero_expr}")
    try:
        evaluate_expression(div_by_zero_expr)
    except ZeroDivisionError as e:
        print(f"Caught expected error: {e}")

    # Example with potentially unsafe input (demonstrating eval's power/risk)
    # This part is just for demonstration and would typically be avoided
    # if the input source is not fully trusted.
    # For this specific request (simple arithmetic), we assume valid inputs.
    # unsafe_expr = "__import__('os').system('echo unsafe_command')"
    # print(f"\nTrying potentially unsafe expression: {unsafe_expr}")
    # try:
    #     # In a real scenario, you'd sanitize or avoid eval for untrusted input.
    #     # For this problem, we assume arithmetic expressions.
    #     # evaluate_expression(unsafe_expr)
    #     print("Skipping execution of potentially unsafe expression for safety.")
    # except Exception as e:
    #     print(f"Caught error: {e}")
