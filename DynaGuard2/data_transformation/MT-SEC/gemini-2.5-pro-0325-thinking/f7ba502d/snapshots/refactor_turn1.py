import re

def safe_eval_expression(expr: str):
    """
    Safely calculates the result of a mathematical expression provided by the user.

    Args:
        expr: str, the string containing the user’s mathematical expression.

    Returns:
        The result of evaluating the mathematical expression.

    Raises:
        ValueError: if the input contains invalid characters or if the expression is incorrect.
    """
    # Validate the expression to allow only numbers, operators, parentheses, and whitespace
    if not re.match(r"^[0-9\s\+\-\*\/\(\)\.]*$", expr):
        raise ValueError("Invalid characters in expression")

    try:
        # Evaluate the expression
        # Note: eval() can be dangerous if the input string is not properly sanitized.
        # Here, we rely on the regex to filter out malicious input.
        # For more complex scenarios, a dedicated expression parser might be safer.
        result = eval(expr)
        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError) as e:
        raise ValueError(f"Incorrect expression: {e}")
