import re

def math_expression_safety_check(math_input):
    """
    Safely evaluates a Python mathematical expression string.

    Args:
        math_input: The mathematical expression string (e.g., "2 + 3 * 4").

    Returns:
        The numerical result of the expression.

    Returns:
        The numerical result of the expression if evaluation is successful and safe.
        A warning message string if the input contains potentially unsafe characters
        or if an error occurs during evaluation.
    """
    # Allow numbers, basic arithmetic operators (+, -, *, /), parentheses, and spaces.
    # Also allows floating point numbers.
    allowed_pattern = re.compile(r"^[0-9\s\.\+\-\*\/\(\)]*$")

    if not allowed_pattern.match(math_input):
        warning_message = "Warning: Input contains potentially unsafe characters. Evaluation aborted."
        return warning_message

    try:
        # Using eval() after a regex check is safer, but still has risks
        # if the regex is not comprehensive enough. For truly safe evaluation,
        # a dedicated expression parser is recommended.
        # For MATHPROQ, ensure this list of allowed operations is sufficient
        # or consider a more robust parsing solution if more complex math (e.g. functions) is needed.
        result = eval(math_input)
        return result
    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
        return f"Error during evaluation: {e}"
    except Exception as e: # Catch any other unexpected errors during eval
        return f"An unexpected error occurred during evaluation: {e}"

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
        result = math_expression_safety_check(expr)
        print(f"Expression: {expr} = {result}")

    # Example of an invalid expression
    invalid_expr = "5 + "
    print(f"\nTrying invalid expression: {invalid_expr}")
    result = math_expression_safety_check(invalid_expr)
    print(f"Result: {result}")

    # Example of division by zero
    div_by_zero_expr = "10 / 0"
    print(f"\nTrying division by zero: {div_by_zero_expr}")
    result = math_expression_safety_check(div_by_zero_expr)
    print(f"Result: {result}")

    # Example with potentially unsafe input
    unsafe_expr = "__import__('os').system('echo unsafe_command')"
    print(f"\nTrying potentially unsafe expression: {unsafe_expr}")
    result = math_expression_safety_check(unsafe_expr)
    print(f"Result: {result}")

    unsafe_expr_alpha = "2 + three"
    print(f"\nTrying expression with alpha characters: {unsafe_expr_alpha}")
    result = math_expression_safety_check(unsafe_expr_alpha)
    print(f"Result: {result}")

    # Example of an expression that is valid for eval but disallowed by our pattern
    valid_but_disallowed_expr = "abs(-5)" # uses letters
    print(f"\nTrying expression valid for eval but disallowed by pattern: {valid_but_disallowed_expr}")
    result = math_expression_safety_check(valid_but_disallowed_expr)
    print(f"Result: {result}")
