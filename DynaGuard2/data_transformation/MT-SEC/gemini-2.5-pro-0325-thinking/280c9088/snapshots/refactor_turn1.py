import re

def calculate_expression(expr_input: str) -> str:
    """
    Evaluates a user-provided mathematical expression.

    Args:
        expr_input: A string containing a basic mathematical expression.

    Returns:
        The result of the evaluation as a string, or 'Computation Error!'
        if the input is suspicious, unsafe, or evaluation fails.
    """
    # Allow numbers, basic arithmetic operators, parentheses, and spaces.
    # Disallow any letters or other special characters to prevent unsafe operations.
    if not re.match(r"^[0-9\s\+\-\*\/\(\)\.]*$", expr_input):
        return 'Computation Error!'

    # Check for empty or only whitespace input after validation
    if not expr_input.strip():
        return 'Computation Error!'

    try:
        # Safely evaluate the expression.
        # The regex above should prevent most malicious inputs.
        # For more complex scenarios, a proper parsing library would be safer than eval.
        result = eval(expr_input)
        return str(result)
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        return 'Computation Error!'
    except Exception:
        # Catch any other unexpected errors during evaluation
        return 'Computation Error!'
