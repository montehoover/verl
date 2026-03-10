import re

def evaluate_expression(expression_string):
    """
    Evaluates a simple arithmetic expression string safely.

    Args:
        expression_string: The arithmetic expression (e.g., "2 + 3 * 4").

    Returns:
        The numerical result of the expression.

    Raises:
        SyntaxError: If the expression is invalid.
        NameError: If the expression contains invalid names.
        TypeError: If operations are attempted on incompatible types.
        ZeroDivisionError: If division by zero is attempted.
        ValueError: If the expression contains unsafe characters.
    """
    # Allow numbers, basic arithmetic operators (+, -, *, /), parentheses, and spaces.
    # Also allows floating point numbers.
    allowed_pattern = re.compile(r"^[0-9\s\.\+\-\*\/\(\)]*$")

    if not allowed_pattern.match(expression_string):
        warning_message = "Warning: Input contains potentially unsafe characters. Evaluation aborted."
        print(warning_message)
        raise ValueError(warning_message)

    try:
        # Using eval() after a regex check is safer, but still has risks
        # if the regex is not comprehensive enough. For truly safe evaluation,
        # a dedicated expression parser is recommended.
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

    # Example with potentially unsafe input
    unsafe_expr = "__import__('os').system('echo unsafe_command')"
    print(f"\nTrying potentially unsafe expression: {unsafe_expr}")
    try:
        evaluate_expression(unsafe_expr)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    unsafe_expr_alpha = "2 + three"
    print(f"\nTrying expression with alpha characters: {unsafe_expr_alpha}")
    try:
        evaluate_expression(unsafe_expr_alpha)
    except ValueError as e:
        print(f"Caught expected error: {e}")
